"""
Factory module for creating different types of encoders.
This is a simplified version for demonstration purposes.
"""

# Import directly in the function to avoid circular imports
# from encoders import EncoderVisionTower

def create_encoder(model_name_or_path, encoder_type=None):
    """
    Create an encoder of the specified type.
    
    Args:
        model_name_or_path (str): The name or path of the model to load.
        encoder_type (str, optional): The type of encoder to create. Default is None.
            Options include:
            - "clip": OpenAI CLIP models
            - "sigclip": Google SigLip models
            - "biomedclip": Microsoft BiomedCLIP models
            - None: Defaults to CLIP
            
    Returns:
        An encoder instance.
    """
    print(f"Creating encoder of type {encoder_type} for model {model_name_or_path}")
    
    # Import here to avoid circular imports
    from encoders import EncoderVisionTower
    
    # Create a default configuration
    class Config:
        def __init__(self):
            self.mm_vision_select_layer = -2
            self.mm_vision_select_feature = 'patch'
    
    config = Config()
    
    # Create the encoder based on type
    if encoder_type == 'clip' or encoder_type is None:
        return EncoderVisionTower(model_name_or_path, config)
    elif encoder_type == 'sigclip' or encoder_type == 'siglip':
        from encoders.sigclip import SigClipEncoder
        # For SigClip, we should check if the model path contains siglip
        # If not, we can suggest a default SigClip model
        if 'siglip' not in model_name_or_path.lower():
            print(f"Warning: Model path does not contain 'siglip'. Using model as provided: {model_name_or_path}")
            print("For best results with sigclip, use models like 'google/siglip-base-patch16-224'")
        return SigClipEncoder(model_name_or_path, config)
    elif encoder_type == 'biomedclip' or encoder_type == 'medclip':
        from encoders.biomedclip import BiomedClipEncoder
        # Check if model path contains biomedclip
        if 'biomedclip' not in model_name_or_path.lower() and 'medclip' not in model_name_or_path.lower():
            print(f"Warning: Model path does not contain 'biomedclip'. Using model as provided: {model_name_or_path}")
            print("For best results with biomedclip, use models like 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'")
        return BiomedClipEncoder(model_name_or_path, config)
    else:
        # Could handle other encoder types here
        print(f"Encoder type {encoder_type} not supported, using CLIP")
        return EncoderVisionTower(model_name_or_path, config) 