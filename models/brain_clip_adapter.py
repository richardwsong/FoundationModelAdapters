import torch
import torch.nn as nn
import numpy as np
from models.helpers import GenericMLP
from models.helpers import (ACTIVATION_DICT, NORM_DICT, WEIGHT_INIT_DICT,
                            get_clones)
import clip
from torchvision.transforms import ToPILImage

class BrainClipAdapter(nn.Module):

    def __init__(
        self,
        clip_model,
        clip_preprocess,
        fmri_adapter,
        eeg_adapter,
        predictors,
        dataset_config,
    ):
        super().__init__()
        self.fmri_adapter = fmri_adapter
        self.eeg_adapter = eeg_adapter
        self.predictors = predictors
        self.dataset_config = dataset_config
        self.clip_model = clip_model
        self.clip_model.visual.float() 
        self.clip_preprocess = clip_preprocess
        for param in self.clip_model.parameters():
            param.requires_grad = False 
        self._reset_parameters()

    def _reset_parameters(self):
        func = WEIGHT_INIT_DICT["xavier_uniform"]
        for p in self.parameters():
            if p.dim() > 1:
                func(p)

    def get_clip_img_feature(self, image):
        image_features = self.clip_model.visual(image.to(self.clip_model.visual.conv1.weight.device).type_as(self.clip_model.visual.conv1.weight))
        return image_features

    def run_fmri_adapter(self, image_feats):
        x = self.fmri_adapter(image_feats)   
        ratio = 0.2
        fmri_feats = ratio * x + (1 - ratio) * image_feats
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * fmri_feats @ fmri_feats.t() 
        return x, fmri_feats, logits

    def run_eeg_adapter(self, image_feats):
        x = self.eeg_adapter(image_feats)
        ratio = 0.2
        eeg_feats = ratio * x + (1 - ratio) * image_feats 
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * eeg_feats @ eeg_feats.t()
        return x, eeg_feats, logits
    
    def get_predictions(self, fmri_feed_feats, sample_num):
        vigilance = self.predictors["vigilance_head"](fmri_feed_feats)
        predicted_label = torch.argmax(vigilance, dim=1)
        predicted_name = predicted_label
        vigilance_result = vigilance.reshape((sample_num, fmri_feed_feats.shape[0]//sample_num, 2))
        predicted_name_result = predicted_name.reshape(sample_num, fmri_feed_feats.shape[0]//sample_num, 1)
        predictions = {
            "vigilance_head_logits": vigilance_result,
            "vigilance_head": predicted_name_result,
        }
        return predictions
    
    def forward(self, inputs, is_train=True):

        input_eeg_index_seq = inputs["eeg_index"]
        input_eeg_index_seq_shape = input_eeg_index_seq.shape
        input_eeg_index_seq = input_eeg_index_seq.reshape(input_eeg_index_seq.shape[0]*input_eeg_index_seq.shape[1])

        input_eeg_spectrogram = inputs["eeg_spectrogram_img"]
        input_fmri_spectrograms = inputs["fmri_spectrogram_imgs"]
        transpose_eeg_spectrogram = input_eeg_spectrogram.transpose(1, 2) 
        scale_ratio_eeg = transpose_eeg_spectrogram.shape[1] // 690
        transpose_fmri_spectrograms = input_fmri_spectrograms.transpose(1, 2) 
        scale_ratio_fmri = transpose_fmri_spectrograms.shape[1] // 690

        sample_num = input_eeg_index_seq_shape[0]
        window_size = 30
        step_size = 15
        eeg_spectrogram_windows = transpose_eeg_spectrogram.unfold(dimension=1, size=window_size*scale_ratio_eeg, step=step_size*scale_ratio_eeg) # ([4, 114, 570, 3, 30])
        eeg_spectrogram_seq = eeg_spectrogram_windows.reshape(sample_num*eeg_spectrogram_windows.shape[1], eeg_spectrogram_windows.shape[2], eeg_spectrogram_windows.shape[3], eeg_spectrogram_windows.shape[4]).permute(0, 2, 3, 1)
        fmri_spectrogram_windows = transpose_fmri_spectrograms.unfold(dimension=1, size=window_size*scale_ratio_fmri, step=step_size*scale_ratio_fmri) # ([4, 114, 570, 3, 30])
        fmri_spectrograms_seq = fmri_spectrogram_windows.reshape(sample_num*fmri_spectrogram_windows.shape[1], fmri_spectrogram_windows.shape[2], fmri_spectrogram_windows.shape[3], fmri_spectrogram_windows.shape[4]).permute(0, 2, 3, 1)

        np.random.seed(0)
        assert len(fmri_spectrograms_seq) == len(eeg_spectrogram_seq) == len(input_eeg_index_seq)
        if is_train == True:
            indices = np.random.permutation(len(fmri_spectrograms_seq))
            fmri_spectrograms = fmri_spectrograms_seq[indices]
            eeg_spectrogram = eeg_spectrogram_seq[indices]
            input_eeg_index = input_eeg_index_seq[indices]
        else:
            fmri_spectrograms = fmri_spectrograms_seq
            eeg_spectrogram = eeg_spectrogram_seq
            input_eeg_index = input_eeg_index_seq
            
        assert hasattr(self.clip_model.visual, "conv1"), "CLIP model not initialized properly."
        eeg_spectrogram_windows_preprocessed = torch.stack([self.clip_preprocess(ToPILImage()(img.byte())) for img in eeg_spectrogram]).to(torch.float32)
        fmri_spectrogram_windows_preprocessed = torch.stack([self.clip_preprocess(ToPILImage()(img.byte())) for img in fmri_spectrograms]).to(torch.float32)

        eeg_spectrogram_windows_preprocessed = eeg_spectrogram_windows_preprocessed.to(self.clip_model.visual.conv1.weight.device).type_as(self.clip_model.visual.conv1.weight)
        eeg_clip_feats = self.clip_model.visual(eeg_spectrogram_windows_preprocessed).to(torch.float32)
        fmri_spectrogram_windows_preprocessed = fmri_spectrogram_windows_preprocessed.to(self.clip_model.visual.conv1.weight.device).type_as(self.clip_model.visual.conv1.weight)
        fmri_clip_feats = self.clip_model.visual(fmri_spectrogram_windows_preprocessed).to(torch.float32)

        eeg_adapted_feats, eeg_fused_feats, eeg_logits = self.run_eeg_adapter(eeg_clip_feats)
        fmri_adapted_feats, fmri_fused_feats, fmri_logits = self.run_fmri_adapter(fmri_clip_feats)
        predictions = self.get_predictions(fmri_fused_feats, sample_num)

        ret_dict = {}
        ret_dict["predictions"] = predictions
        ret_dict["eeg_index"] = input_eeg_index.reshape(input_eeg_index_seq_shape)
        ret_dict["fmri_feats"] = fmri_fused_feats
        return ret_dict


def build_CLIP(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess


def build_fmri_adapter(args):
    c_in =512
    reduction = 2
    fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )
    return fc


def build_eeg_adapter(args):
    c_in =512
    reduction = 2
    fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )
    return fc
    

def build_predictors(args, dataset_config):
    vigilance_cls_head = GenericMLP(
            input_dim=512   ,
            hidden_dims=[64, 32, 16],
            output_dim=2,
            norm_fn_name="bn1d",
            activation="leakyrelu",
            use_conv=False,
            dropout=0.2,
            hidden_use_bias=True,
        )
    predictors = [
        ("vigilance_head", vigilance_cls_head),
    ]
    return nn.ModuleDict(predictors)


def build_brain_clip_adapter(args, dataset_config):
    clip_model, clip_preprocess = build_CLIP(args)
    fmri_adapter = build_fmri_adapter(args)
    eeg_adapter = build_eeg_adapter(args)
    predictors = build_predictors(args, dataset_config)
    model = BrainClipAdapter(
        clip_model,
        clip_preprocess,
        fmri_adapter,
        eeg_adapter,
        predictors,
        dataset_config,
    )
    return model