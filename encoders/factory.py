from .base import ImageEncoder
from .clip import ClipEncoder
from .sigclip import SigClipEncoder

def create_encoder(encoder_type, **kwargs):
    """
    Factory function to create image encoders.
    
    Args:
        encoder_type: String identifier for encoder type ('clip', 'sigclip', 'vit', etc.)
        **kwargs: Additional arguments to pass to the encoder constructor
        
    Returns:
        An instance of the requested encoder
    """
    encoder_type = encoder_type.lower()
    
    if encoder_type == "clip":
        return ClipEncoder(**kwargs)
    elif encoder_type == "sigclip":
        return SigClipEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}") 