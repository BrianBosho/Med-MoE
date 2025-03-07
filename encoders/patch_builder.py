"""
This is a patch for the moellava/model/multimodal_encoder/builder.py file
to integrate our encoder framework with the existing codebase.

Instructions:
1. Locate moellava/model/multimodal_encoder/builder.py in your codebase
2. Add the imports and modify the build_vision_tower function as shown below
"""

# Original imports
# from .clip_encoder import CLIPVisionTower
# from .siglip_encoder import SigLipVisionTower

# Add this import
from encoders import EncoderVisionTower

def build_vision_tower(image_tower, args, **kwargs):
    """
    Build a vision tower.
    
    Args:
        image_tower (str): The name of the vision tower.
        args: The arguments for building the vision tower.
        
    Returns:
        A vision tower.
    """
    image_tower_cfg = getattr(args, 'mm_vision_tower_cfg', None)
    
    # First check if we should use the new encoder framework
    # This can be enabled via a flag in the args
    if hasattr(args, 'use_encoder_framework') and args.use_encoder_framework:
        print(f"Using encoder framework for {image_tower}")
        return EncoderVisionTower(image_tower, args=args, **kwargs)
    
    # Otherwise use the original logic
    if 'clip-vit' in image_tower.lower():
        from .clip_encoder import CLIPVisionTower
        return CLIPVisionTower(image_tower, args=image_tower_cfg, **kwargs)
    elif 'siglip' in image_tower.lower():
        from .siglip_encoder import SigLipVisionTower
        return SigLipVisionTower(image_tower, args=image_tower_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown image tower: {image_tower}') 