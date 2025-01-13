from .brain_clip_adapter import build_brain_clip_adapter

MODEL_FUNCS = {
    "brain_clip_adapter": build_brain_clip_adapter,
}

def build_model(args, dataset_config):
    model = MODEL_FUNCS[args.model_name](args, dataset_config)
    return model