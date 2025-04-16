from torch import nn
import numpy as np

class SetCriterion(nn.Module):
    def __init__(self, dataset_config, loss_weight_dict, class_weights=None):
        super().__init__()
        self.dataset_config = dataset_config
        self.loss_weight_dict = loss_weight_dict
        self.class_weights = class_weights

        self.loss_functions = {
            "loss_fmri_prediction": self.loss_fmri_prediction,
        }

    def loss_fmri_prediction(self, outputs, targets):
        loss_prediction_criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        y_true = targets["vigilance_label"].long().view(-1)  # shape [B]
        y_pred = outputs["predictions"]["vigilance_head_logits"]  # [B, T, C]
        y_pred = y_pred.mean(dim=1)  # Now [B, C]
        curr_loss = loss_prediction_criterion(y_pred, y_true)
        return {"loss_fmri_prediction": curr_loss}

    def single_output_forward(self, outputs, targets):
        losses = {}
        for f in self.loss_functions:
            loss_wt_key = f + "_weight"
            if (
                loss_wt_key in self.loss_weight_dict
                and self.loss_weight_dict[loss_wt_key] > 0
            ) or loss_wt_key not in self.loss_weight_dict:
                curr_loss = self.loss_functions[f](outputs, targets)
                losses.update(curr_loss)

        final_loss = 0.0
        for w in self.loss_weight_dict:
            if self.loss_weight_dict[w] > 0:
                losses[w.replace("_weight", "")] *= self.loss_weight_dict[w]
                final_loss += losses[w.replace("_weight", "")]
        return final_loss, losses

    def forward(self, outputs, targets):
        loss, loss_dict = self.single_output_forward(outputs, targets)
        return loss, loss_dict


def build_criterion(args, dataset_config):
    from collections import Counter
    import torch
    from datasets.SampleDataset import SampleDataset, SampleDatasetConfig

    # Count class balance from training set
    train_dataset = SampleDataset(SampleDatasetConfig(), split_set="train")
    labels = [sample["vigilance_label"] for sample in train_dataset]

    label_counts = Counter(labels)
    total = sum(label_counts.values())
    weights = torch.tensor(
        [total / label_counts.get(i, 1) for i in range(2)], dtype=torch.float32
    )
    weights = weights / weights.sum()

    loss_weight_dict = {
        "loss_fmri_prediction_weight": args.loss_fmri_prediction_weight,
    }
    criterion = SetCriterion(dataset_config, loss_weight_dict, class_weights=weights.to(args.train_device))
    return criterion
