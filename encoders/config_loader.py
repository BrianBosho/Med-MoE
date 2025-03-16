"""
Configuration loader for the encoders framework.
"""
import os
import yaml
import argparse
from types import SimpleNamespace

DEFAULT_CONFIG = {
    'image_encoder': {
        'type': 'clip',
        'model_name': 'openai/clip-vit-large-patch14',
        'select_layer': -1,
        'select_feature': 'patch',
        'use_encoder_framework': True
    },
    'video_encoder': {
        'type': 'none',
        'model_name': None,
        'select_layer': -1,
        'select_feature': 'patch'
    },
    'projector': {
        'image_projector_type': 'linear',
        'mm_hidden_size': 1024,
        'hidden_size': 4096
    },
    'cache_dir': './cache_dir',
    'precision': 'float16'
}

def load_config(config_path=None):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file. If None, returns default config.
        
    Returns:
        SimpleNamespace object with configuration attributes
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            
        # Update the default config with values from the YAML file
        for section in config:
            if section in yaml_config:
                if isinstance(config[section], dict) and isinstance(yaml_config[section], dict):
                    config[section].update(yaml_config[section])
                else:
                    config[section] = yaml_config[section]
    
    # Convert to SimpleNamespace for attribute-style access
    return _dict_to_namespace(config)

def _dict_to_namespace(d):
    """
    Convert a dictionary to a SimpleNamespace, recursively.
    
    Args:
        d: Dictionary to convert
        
    Returns:
        SimpleNamespace with the same structure as the dictionary
    """
    if isinstance(d, dict):
        # Convert all items in the dictionary
        for key, value in d.items():
            d[key] = _dict_to_namespace(value)
        return SimpleNamespace(**d)
    elif isinstance(d, list):
        # Convert all items in the list
        return [_dict_to_namespace(item) for item in d]
    else:
        # Return the value as is
        return d

def config_to_args(config):
    """
    Convert a configuration object to an argparse.Namespace object
    that can be used with the existing Med-MoE codebase.
    
    Args:
        config: Configuration object (SimpleNamespace)
        
    Returns:
        argparse.Namespace object
    """
    args = argparse.Namespace()
    
    # Image encoder configuration
    args.mm_vision_select_layer = config.image_encoder.select_layer
    args.mm_vision_select_feature = config.image_encoder.select_feature
    args.use_encoder_framework = config.image_encoder.use_encoder_framework
    
    # Set image tower to the model name
    if config.image_encoder.type != 'none':
        args.image_tower = config.image_encoder.model_name
    else:
        args.image_tower = None
    
    # Video encoder configuration
    if config.video_encoder.type != 'none':
        args.video_tower = config.video_encoder.model_name
    else:
        args.video_tower = None
    
    # Projector configuration
    args.image_projector_type = config.projector.image_projector_type
    args.mm_hidden_size = config.projector.mm_hidden_size
    args.hidden_size = config.projector.hidden_size
    
    # Other configurations
    args.cache_dir = config.cache_dir
    
    return args

# Add a helper function to get configs from command line arguments
def get_config_from_args():
    """Parse command line arguments for configuration file."""
    parser = argparse.ArgumentParser(description="Load encoder configuration")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    args = parser.parse_args()
    
    return load_config(args.config)

if __name__ == "__main__":
    # Test the configuration loader
    config = load_config()
    print("Default configuration:")
    print(f"Image encoder type: {config.image_encoder.type}")
    print(f"Image encoder model: {config.image_encoder.model_name}")
    print(f"Projector type: {config.projector.image_projector_type}")
    
    # Convert to args
    args = config_to_args(config)
    print("\nConverted to args:")
    print(f"image_tower: {args.image_tower}")
    print(f"mm_vision_select_layer: {args.mm_vision_select_layer}")
    print(f"image_projector_type: {args.image_projector_type}") 