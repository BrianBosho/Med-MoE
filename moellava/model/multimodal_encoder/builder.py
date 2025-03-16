import os
from .clip_encoder import CLIPVisionTower
import transformers
a, b, c = transformers.__version__.split('.')[:3]
if a == '4' and int(b) >= 37:
    from .siglip_encoder import SiglipVisionTower
# from .languagebind import LanguageBindImageTower, LanguageBindVideoTower

# ============================================================================================================

def build_image_tower(image_tower_cfg, **kwargs):
    """
    Builds and returns an image tower model based on the provided configuration.
    
    Args:
        image_tower_cfg: Configuration object containing image tower settings
        **kwargs: Additional keyword arguments passed to the tower constructors
        
    Returns:
        An instance of the appropriate vision tower class (CLIP, SigLip, or LanguageBind)
        
    Raises:
        ValueError: If the specified image tower type is not recognized
    """
    # Extract image tower name from config, checking both mm_image_tower and image_tower attributes
    image_tower = getattr(image_tower_cfg, 'mm_image_tower', getattr(image_tower_cfg, 'image_tower', None))
    
    # Check if the image_tower string is a valid file path
    is_absolute_path_exists = os.path.exists(image_tower)
    
    # Handle CLIP model IDs by adding appropriate prefix if needed
    # This ensures compatibility with HuggingFace model hub naming conventions
    if not is_absolute_path_exists and not any(image_tower.startswith(prefix) for prefix in ['openai/', 'laion/']):
        if 'clip-vit' in image_tower.lower():
            image_tower = f'openai/{image_tower}'
    
    # Return appropriate vision tower based on the model type/prefix
    # For CLIP models (from OpenAI or LAION) or local checkpoint paths
    if is_absolute_path_exists or image_tower.startswith("openai") or image_tower.startswith("laion"):
        return CLIPVisionTower(image_tower, args=image_tower_cfg, **kwargs)
    
    # For Google's SigLip models
    if image_tower.startswith("google") or ('siglip' in image_tower.lower()):
        print('Using SigLip')
        return SiglipVisionTower(image_tower, image_tower_cfg, **kwargs)
    
    # For LanguageBind image models
    if image_tower.endswith('LanguageBind_Image'):
        return LanguageBindImageTower(image_tower, args=image_tower_cfg, cache_dir='./cache_dir', **kwargs)
    
    raise ValueError(f'Unknown image tower: {image_tower}')

def build_video_tower(video_tower_cfg, **kwargs):
    """
    Builds and returns a video tower model based on the provided configuration.
    
    Args:
        video_tower_cfg: Configuration object containing video tower settings
        **kwargs: Additional keyword arguments passed to the tower constructor
        
    Returns:
        An instance of the appropriate video tower class (currently only LanguageBind)
        
    Raises:
        ValueError: If the specified video tower type is not recognized
    """
    # Extract video tower name from config, checking both mm_video_tower and video_tower attributes
    video_tower = getattr(video_tower_cfg, 'mm_video_tower', getattr(video_tower_cfg, 'video_tower', None))
    
    # Currently only supports LanguageBind video models
    if video_tower.endswith('LanguageBind_Video_merge'):
        return LanguageBindVideoTower(video_tower, args=video_tower_cfg, cache_dir='./cache_dir', **kwargs)
    
    raise ValueError(f'Unknown video tower: {video_tower}')
# ============================================================================================================
