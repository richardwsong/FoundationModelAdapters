from models.brain_clip_adapter import build_brain_clip_adapter
from models.brain_cnn_adapter import build_brain_cnn_adapter  # Add this import

MODEL_FUNCS = {
    "brain_clip_adapter": build_brain_clip_adapter,
    "brain_cnn_adapter": build_brain_cnn_adapter,  # Add this line
}

def build_model(args, dataset_config):
    model = MODEL_FUNCS[args.model_name](args, dataset_config)
    return model