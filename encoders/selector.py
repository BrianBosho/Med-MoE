"""
Encoder selector for MoELLaVA - provides a simple interface for choosing different
vision encoders without changing any other code.
"""

from moellava.model.builder import load_pretrained_model

# Map of friendly names to encoder types
ENCODER_TYPES = {
    "clip": "clip",              # OpenAI CLIP models
    "sigclip": "sigclip",        # Google SigLip models
    "medclip": "medclip",        # Medical CLIP models 
    # Add more encoder types as they become available
}

# Map of encoder models to their image sizes
# This helps match the right image size for each model
ENCODER_IMAGE_SIZES = {
    # CLIP models
    "openai/clip-vit-base-patch16-224": 224,
    "openai/clip-vit-large-patch14-336": 336,
    "openai/clip-vit-large-patch14": 224,
    # SigLip models
    "google/siglip-base-patch16-224": 224,
    "google/siglip-large-patch16-384": 384,
}

def load_model_with_encoder(
    model_path,
    model_base=None,
    model_name=None,
    encoder="clip",  # Default to CLIP
    encoder_path=None,  # Override vision tower path
    device="cuda",
    load_8bit=False,
    load_4bit=False,
    **kwargs
):
    """
    Load a MoELLaVA model with the specified encoder.
    
    Args:
        model_path: Path to the model
        model_base: Optional base model
        model_name: Optional model name
        encoder: Encoder type to use - one of: "clip", "sigclip", "medclip", etc.
        encoder_path: Optional path to the encoder model (overrides default)
        device: Device to load the model on
        load_8bit: Whether to load the model in 8-bit precision
        load_4bit: Whether to load the model in 4-bit precision
        **kwargs: Additional arguments to pass to load_pretrained_model
        
    Returns:
        tokenizer, model, processor, context_len
    """
    # Validate encoder type
    if encoder not in ENCODER_TYPES and encoder not in ENCODER_TYPES.values():
        valid_encoders = ", ".join(ENCODER_TYPES.keys())
        raise ValueError(f"Unknown encoder type: {encoder}. Valid options are: {valid_encoders}")
    
    # Get the actual encoder type
    encoder_type = ENCODER_TYPES.get(encoder, encoder)
    
    # Add encoder type to kwargs
    kwargs["encoder_type"] = encoder_type
    
    # If model_name is not provided, extract it from model_path
    if model_name is None and model_path is not None:
        import os
        model_name = os.path.basename(model_path.rstrip('/'))
        print(f"Extracted model name: {model_name} from path")
    
    # If a custom encoder path is provided, we need to handle that separately
    # by setting the mm_vision_tower parameter
    if encoder_path is not None:
        # Create a custom config with the specified vision tower
        # This will be used in the model loading process
        class CustomConfig:
            def __init__(self):
                self.mm_vision_tower = encoder_path
                self.encoder_type = encoder_type
                
                # Set image size based on encoder model if available
                if encoder_path in ENCODER_IMAGE_SIZES:
                    self.image_size = ENCODER_IMAGE_SIZES[encoder_path]
                    print(f"Setting image size to {self.image_size} for {encoder_path}")
        
        kwargs["custom_vision_config"] = CustomConfig()
    
    print(f"Loading model with encoder: {encoder} (type: {encoder_type})")
    
    # Load the model with the specified encoder
    return load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name,
        load_8bit=load_8bit,
        load_4bit=load_4bit,
        device=device,
        **kwargs
    ) 