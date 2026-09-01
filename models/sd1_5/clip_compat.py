def get_clip_encoder_layers(text_encoder):
    """Return CLIP encoder layers across Transformers 4.x/5.x layouts."""
    if hasattr(text_encoder, "text_model"):
        return text_encoder.text_model.encoder.layers
    return text_encoder.encoder.layers


def get_clip_final_layer_norm(text_encoder):
    """Return CLIP final layer norm across Transformers 4.x/5.x layouts."""
    if hasattr(text_encoder, "text_model"):
        return text_encoder.text_model.final_layer_norm
    return text_encoder.final_layer_norm
