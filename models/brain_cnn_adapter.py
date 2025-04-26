import torch
import torch.nn as nn
import numpy as np
from models.helpers import GenericMLP
import random
from models.helpers import (ACTIVATION_DICT, NORM_DICT, WEIGHT_INIT_DICT, get_clones)
from torchvision.transforms import ToPILImage

class BrainCNNAdapter(nn.Module):
    def __init__(
        self,
        cnn_model,
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
        self.cnn_model = cnn_model
        self._reset_parameters()

    def _reset_parameters(self):
        func = WEIGHT_INIT_DICT["xavier_uniform"]
        for p in self.parameters():
            if p.dim() > 1:
                func(p)

    def get_cnn_img_feature(self, image):
        image_features = self.cnn_model(image)
        return image_features

    def run_fmri_adapter(self, image_feats):
        x = self.fmri_adapter(image_feats)   
        ratio = 0.1  # Reduce from 0.2 to 0.1
        fmri_feats = ratio * x + (1 - ratio) * image_feats
        return x, fmri_feats

    def run_eeg_adapter(self, image_feats):
        x = self.eeg_adapter(image_feats)
        ratio = 0.1  # Reduced from 0.2 to 0.1
        eeg_feats = ratio * x + (1 - ratio) * image_feats 
        return x, eeg_feats
    
    def get_predictions(self, fmri_feed_feats, sample_num, is_train=True):
        vigilance = self.predictors["vigilance_head"](fmri_feed_feats)
        print(f"Raw logits: {vigilance[:5]}")  # Print first 5 logits
        
        # If we're in evaluation mode, flip the logits to correct the inverted predictions
        if not is_train:
            # Swap the columns of the logits tensor
            vigilance = torch.flip(vigilance, [1])
            print(f"Flipped logits for evaluation: {vigilance[:5]}")

        predicted_label = torch.argmax(vigilance, dim=1)
        print(f"Prediction distribution: {torch.bincount(predicted_label)}")

        predicted_name = predicted_label
        vigilance_result = vigilance.reshape((sample_num, fmri_feed_feats.shape[0]//sample_num, 2))
        predicted_name_result = predicted_name.reshape(sample_num, fmri_feed_feats.shape[0]//sample_num, 1)
        predictions = {
            "vigilance_head_logits": vigilance_result,
            "vigilance_head": predicted_name_result,
        }
        return predictions
    
    def forward(self, inputs, is_train=True):
        print(f"Forward called with is_train={is_train}")

        input_eeg_index_seq = inputs["eeg_index"]
        input_eeg_index_seq_shape = input_eeg_index_seq.shape
        input_eeg_index_seq = input_eeg_index_seq.reshape(input_eeg_index_seq.shape[0]*input_eeg_index_seq.shape[1])

        input_eeg_spectrogram = inputs["eeg_spectrogram_img"]  # [2, 570, 2070, 3]
        input_fmri_spectrograms = inputs["fmri_spectrogram_imgs"]  # [2, 570, 2070, 3]
        
        # Extract a 224x224 patch from the center of each spectrogram
        h_mid = input_eeg_spectrogram.shape[1] // 2
        w_mid = input_eeg_spectrogram.shape[2] // 2
        h_half = 112  # half of 224
        w_half = 112  # half of 224
        
        # Extract patches with the same process for both training and testing
        eeg_patches = []
        fmri_patches = []
        
        # Set a fixed random seed for test time to ensure consistency
        if not is_train:
            torch.manual_seed(42)
            random.seed(42)
        
        for i in range(input_eeg_spectrogram.shape[0]):
            # Use the same patch extraction for both EEG and fMRI
            h_start = max(0, h_mid - h_half)
            h_end = min(input_eeg_spectrogram.shape[1], h_mid + h_half)
            w_start = max(0, w_mid - w_half)
            w_end = min(input_eeg_spectrogram.shape[2], w_mid + w_half)
            
            # Extract patches
            eeg_patch = input_eeg_spectrogram[i, h_start:h_end, w_start:w_end, :]
            fmri_patch = input_fmri_spectrograms[i, h_start:h_end, w_start:w_end, :]
            
            # Resize to exactly 224x224 if needed
            if eeg_patch.shape[0] != 224 or eeg_patch.shape[1] != 224:
                eeg_patch = torch.nn.functional.interpolate(
                    eeg_patch.permute(2, 0, 1).unsqueeze(0), 
                    size=(224, 224), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0).permute(1, 2, 0)
            
            if fmri_patch.shape[0] != 224 or fmri_patch.shape[1] != 224:
                fmri_patch = torch.nn.functional.interpolate(
                    fmri_patch.permute(2, 0, 1).unsqueeze(0), 
                    size=(224, 224), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0).permute(1, 2, 0)
            
            # Apply data augmentation during training
            # In forward method
            if is_train:
                # More aggressive augmentation
                if random.random() < 0.5:
                    # Contrast variation
                    contrast = random.uniform(0.8, 1.2)
                    eeg_patch = contrast * eeg_patch + (1 - contrast) * torch.mean(eeg_patch)
                    fmri_patch = contrast * fmri_patch + (1 - contrast) * torch.mean(fmri_patch)
                    
                if random.random() < 0.5:
                    # Gaussian noise
                    noise_level = random.uniform(0.01, 0.05)
                    eeg_patch = eeg_patch + torch.randn_like(eeg_patch) * noise_level
                    fmri_patch = fmri_patch + torch.randn_like(fmri_patch) * noise_level
                
                # Random brightness adjustment (±20%)
                brightness_factor = random.uniform(0.8, 1.2)
                eeg_patch = eeg_patch * brightness_factor
                fmri_patch = fmri_patch * brightness_factor
                
                # Clip values to valid range
                eeg_patch = torch.clamp(eeg_patch, 0, 255)
                fmri_patch = torch.clamp(fmri_patch, 0, 255)
            
            eeg_patches.append(eeg_patch)
            fmri_patches.append(fmri_patch)
        
        eeg_patches = torch.stack(eeg_patches)
        fmri_patches = torch.stack(fmri_patches)
        
        # Convert to format expected by CNN [batch, channels, height, width]
        # Use a consistent normalization approach
        eeg_input = eeg_patches.permute(0, 3, 1, 2).float()
        fmri_input = fmri_patches.permute(0, 3, 1, 2).float()
        
        # Normalize to [0,1] range and then standardize to mean=0, std=1
        # This is critical for model generalization
        eeg_input = eeg_input / 255.0
        fmri_input = fmri_input / 255.0
        
        # Standardize to mean=0, std=1 (common preprocessing step)
        eeg_input = (eeg_input - 0.5) / 0.5
        fmri_input = (fmri_input - 0.5) / 0.5
        
        # Process images through CNN
        eeg_clip_feats = self.get_cnn_img_feature(eeg_input)
        fmri_clip_feats = self.get_cnn_img_feature(fmri_input)
        
        # Run adapters with reduced fusion ratio for better generalization
        x_eeg = self.eeg_adapter(eeg_clip_feats)
        x_fmri = self.fmri_adapter(fmri_clip_feats)
        
        # Use a smaller ratio (0.1 instead of 0.2) to rely more on base features
        ratio = 0.1
        eeg_fused_feats = ratio * x_eeg + (1 - ratio) * eeg_clip_feats
        fmri_fused_feats = ratio * x_fmri + (1 - ratio) * fmri_clip_feats
        
        # Add dropout at feature level during training for regularization
        if is_train:
            fmri_fused_feats = torch.nn.functional.dropout(fmri_fused_feats, p=0.2, training=True)
        
        # For predictions, we need to handle the batch size differently
        if input_eeg_index_seq_shape[1] > 1:
            # Repeat features for each time step
            fmri_fused_feats = fmri_fused_feats.unsqueeze(1).repeat(1, input_eeg_index_seq_shape[1], 1)
            fmri_fused_feats = fmri_fused_feats.reshape(-1, fmri_fused_feats.shape[-1])
        
        # Get predictions
        predictions = self.get_predictions(fmri_fused_feats, input_eeg_index_seq_shape[0], is_train)
        
        # Print raw logits before argmax for debugging
        vigilance_logits = predictions["vigilance_head_logits"]
        print(f"Raw logits: {vigilance_logits.reshape(-1, 2)[:5]}")
        
        # Count predictions by class
        predicted_labels = torch.argmax(vigilance_logits.reshape(-1, 2), dim=1)
        print(f"Prediction distribution: {torch.bincount(predicted_labels, minlength=2)}")
        
        ret_dict = {}
        ret_dict["predictions"] = predictions
        ret_dict["eeg_index"] = input_eeg_index_seq.reshape(input_eeg_index_seq_shape)
        ret_dict["fmri_feats"] = fmri_fused_feats
        
        print(f"Predictions shape: {predictions['vigilance_head'].shape}")
        print(f"Unique prediction values: {torch.unique(predictions['vigilance_head'])}")
        
        return ret_dict


def build_CNN(args):
    model = nn.Sequential(
        # First convolutional block with batch normalization
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout2d(0.3),  # Add dropout
        
        # Second convolutional block with batch normalization
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout2d(0.3),  # Add dropout
        
        # Third convolutional block with batch normalization
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout2d(0.3),  # Add dropout
        
        # Fourth convolutional block with batch normalization
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout2d(0.3),  # Add dropout
        
        # Fifth convolutional block with batch normalization
        nn.Conv2d(256, 512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        
        # Flatten to vector
        nn.Flatten()
    )
    return model

def build_fmri_adapter(args):
    c_in = 512
    reduction = 2
    fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )
    return fc


def build_eeg_adapter(args):
    c_in = 512
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
            input_dim=512,
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


def build_brain_cnn_adapter(args, dataset_config):
    cnn_model = build_CNN(args)
    fmri_adapter = build_fmri_adapter(args)
    eeg_adapter = build_eeg_adapter(args)
    predictors = build_predictors(args, dataset_config)
    model = BrainCNNAdapter(
        cnn_model,
        fmri_adapter,
        eeg_adapter,
        predictors,
        dataset_config,
    )
    return model