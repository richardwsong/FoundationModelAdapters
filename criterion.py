from torch import nn
import numpy as np
import torch

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
        # Use a stronger class weighting - even if you already have weights
        # This will put more emphasis on the minority class
        # Assuming class 1 is the minority class in the test set
        custom_weights = torch.tensor([0.3, 0.7], device=targets["vigilance_label"].device)
        
        # Combine with existing weights if available
        if self.class_weights is not None:
            effective_weights = self.class_weights * custom_weights
            effective_weights = effective_weights / effective_weights.sum()
        else:
            effective_weights = custom_weights
            
        loss_prediction_criterion = nn.CrossEntropyLoss(weight=effective_weights)
        
        y_true = targets["vigilance_label"].long().view(-1)  # shape [B]
        y_pred = outputs["predictions"]["vigilance_head_logits"]  # [B, T, C]
        y_pred = y_pred.mean(dim=1)  # Now [B, C]
        
        # Add label smoothing to improve generalization
        smoothing = 0.1
        if smoothing > 0:
            n_classes = y_pred.size(1)
            y_hot = torch.zeros_like(y_pred).scatter_(1, y_true.unsqueeze(1), 1)
            y_smooth = y_hot * (1 - smoothing) + smoothing / n_classes
            
            # Manual cross entropy with label smoothing
            log_probs = torch.nn.functional.log_softmax(y_pred, dim=1)
            loss = -(y_smooth * log_probs).sum(dim=1)
            loss = loss.mean()
        else:
            # Standard cross entropy with weights
            loss = loss_prediction_criterion(y_pred, y_true)
        
        return {"loss_fmri_prediction": loss}

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

    
    # Compute inverse frequency weights
    weights = torch.tensor(
        [total / label_counts.get(i, 1) for i in range(2)], dtype=torch.float32
    )
    
    # Optional: Make weights more extreme to focus more on minority class
    # weights = weights ** 1.5  # Raising to power > 1 increases contrast
    
    # Normalize weights to sum to 1
    weights = weights / weights.sum()

    loss_weight_dict = {
        "loss_fmri_prediction_weight": args.loss_fmri_prediction_weight,
    }
    criterion = SetCriterion(dataset_config, loss_weight_dict, class_weights=weights.to(args.train_device))
    return criterion