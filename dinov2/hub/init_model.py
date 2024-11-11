from dinov2.hub.backbones import (
    dinov2_vitb14,
    dinov2_vitg14,
    dinov2_vitl14,
    dinov2_vits14,
)


def initialize_model(model_name: str, pretrained: bool = True, **kwargs):
    """
    Initializes the specified DINOv2 model.

    Args:
        model_name (str): Name of the model to initialize (e.g., 'vitb14', 'vitg14').
        pretrained (bool): Whether to load pretrained weights.
        **kwargs: Additional keyword arguments for model initialization.

    Returns:
        torch.nn.Module: The initialized model.
    """
    model_dict = {
        "vitb14": dinov2_vitb14,
        "vitg14": dinov2_vitg14,
        "vitl14": dinov2_vitl14,
        "vits14": dinov2_vits14,
    }

    if model_name not in model_dict:
        raise ValueError(
            f"Unsupported model name '{model_name}'. Available models: {list(model_dict.keys())}"
        )

    model = model_dict[model_name](pretrained=pretrained, **kwargs)
    return model
