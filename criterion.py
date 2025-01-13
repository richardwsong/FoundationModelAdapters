from torch import nn

class SetCriterion(nn.Module):
    def __init__(self, dataset_config, loss_weight_dict):
        super().__init__()
        self.dataset_config = dataset_config
        self.loss_weight_dict = loss_weight_dict

        self.loss_functions = {
            "loss_fmri_prediction": self.loss_fmri_prediction,
        }
        
    def loss_fmri_prediction(self, outputs, targets):
        loss_prediction_criterion = nn.CrossEntropyLoss()
        y_true = targets["eeg_index"].long()
        y_pred = outputs["predictions"]["vigilance_head_logits"]
        y_true_reshape = y_true.reshape(y_true.shape[0], y_true.shape[1])
        curr_loss = loss_prediction_criterion(y_pred.transpose(1, 2), y_true_reshape)
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
    loss_weight_dict = {
        "loss_fmri_prediction_weight": args.loss_fmri_prediction_weight,
    }
    criterion = SetCriterion(dataset_config, loss_weight_dict)
    return criterion