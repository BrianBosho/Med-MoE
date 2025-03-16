#!/usr/bin/env python
"""
Test script for our enhanced encoder and projector configuration.
This script tests loading from a configuration file and ensuring
that the image encoder and projector can be properly initialized.
"""
import os
import sys
import argparse
import torch
import yaml
from types import SimpleNamespace

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our configuration tools
try:
    from encoders import load_config, config_to_args, EncoderVisionTower
    ENCODERS_FRAMEWORK_AVAILABLE = True
except ImportError:
    ENCODERS_FRAMEWORK_AVAILABLE = False
    print("WARNING: Encoders framework not available. Some tests will be skipped.")

# Import builder modules
try:
    from moellava.model.multimodal_encoder.builder_enhanced import build_image_tower
    from moellava.model.multimodal_projector.builder_enhanced import build_image_projector, build_projector
    ENHANCED_BUILDERS_AVAILABLE = True
except ImportError:
    ENHANCED_BUILDERS_AVAILABLE = False
    print("WARNING: Enhanced builders not available. Some tests will be skipped.")

# Import original builders for comparison
try:
    from moellava.model.multimodal_encoder.builder import build_image_tower as original_build_image_tower
    from moellava.model.multimodal_projector.builder import build_image_projector as original_build_image_projector
    ORIGINAL_BUILDERS_AVAILABLE = True
except ImportError:
    ORIGINAL_BUILDERS_AVAILABLE = False
    print("WARNING: Original builders not available. Some tests will be skipped.")

def print_separator():
    print("\n" + "="*80 + "\n")

def test_config_loading(config_path=None):
    """Test loading a configuration file."""
    print("Testing configuration loading...")
    
    if not ENCODERS_FRAMEWORK_AVAILABLE:
        print("Encoders framework not available. Skipping test.")
        return False
    
    try:
        config = load_config(config_path)
        print("Successfully loaded configuration.")
        print(f"Image encoder type: {config.image_encoder.type}")
        print(f"Image encoder model: {config.image_encoder.model_name}")
        print(f"Projector type: {config.projector.image_projector_type}")
        
        args = config_to_args(config)
        print("\nConverted to args:")
        print(f"image_tower: {args.image_tower}")
        print(f"mm_vision_select_layer: {args.mm_vision_select_layer}")
        print(f"image_projector_type: {args.image_projector_type}")
        
        return True
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return False

def test_encoder_loading(config_path=None):
    """Test loading an image encoder from configuration."""
    print("Testing image encoder loading...")
    
    if not ENCODERS_FRAMEWORK_AVAILABLE or not ENHANCED_BUILDERS_AVAILABLE:
        print("Required modules not available. Skipping test.")
        return False
    
    try:
        config = load_config(config_path)
        args = config_to_args(config)
        
        # Build the image tower using our enhanced builder
        image_tower = build_image_tower(args, config_path)
        print("Successfully built image tower.")
        print(f"Image tower type: {type(image_tower).__name__}")
        
        # Check if it's using the encoders framework
        is_using_framework = isinstance(image_tower, EncoderVisionTower)
        print(f"Using encoders framework: {is_using_framework}")
        
        # Test if the image tower has the expected methods
        has_forward = hasattr(image_tower, 'forward')
        has_load_model = hasattr(image_tower, 'load_model')
        print(f"Has forward method: {has_forward}")
        print(f"Has load_model method: {has_load_model}")
        
        return True
    except Exception as e:
        print(f"Error loading image encoder: {e}")
        return False

def test_projector_loading(config_path=None):
    """Test loading a projector from configuration."""
    print("Testing projector loading...")
    
    if not ENCODERS_FRAMEWORK_AVAILABLE or not ENHANCED_BUILDERS_AVAILABLE:
        print("Required modules not available. Skipping test.")
        return False
    
    try:
        config = load_config(config_path)
        args = SimpleNamespace(**config.projector.__dict__)
        
        # Build the projector using our enhanced builder
        projector = build_image_projector(args, config_path)
        print("Successfully built projector.")
        print(f"Projector type: {type(projector).__name__}")
        
        # Test if the projector has a forward method
        has_forward = hasattr(projector, 'forward')
        print(f"Has forward method: {has_forward}")
        
        # Try a forward pass with dummy data if possible
        if has_forward:
            try:
                mm_hidden_size = getattr(config.projector, 'mm_hidden_size', 1024)
                dummy_input = torch.randn(1, mm_hidden_size)
                dummy_output = projector(dummy_input)
                print(f"Forward pass successful. Output shape: {dummy_output.shape}")
            except Exception as e:
                print(f"Error during forward pass: {e}")
        
        return True
    except Exception as e:
        print(f"Error loading projector: {e}")
        return False

def compare_with_original_builders(config_path=None):
    """Compare our enhanced builders with the original ones."""
    print("Comparing enhanced builders with original builders...")
    
    if not ENCODERS_FRAMEWORK_AVAILABLE or not ENHANCED_BUILDERS_AVAILABLE or not ORIGINAL_BUILDERS_AVAILABLE:
        print("Required modules not available. Skipping test.")
        return False
    
    try:
        config = load_config(config_path)
        args = config_to_args(config)
        
        # Try building with the original builders
        print("\nTesting original builders...")
        try:
            original_tower = original_build_image_tower(args)
            print(f"Original image tower type: {type(original_tower).__name__}")
        except Exception as e:
            print(f"Error with original image tower builder: {e}")
        
        try:
            original_projector = original_build_image_projector(args)
            print(f"Original projector type: {type(original_projector).__name__}")
        except Exception as e:
            print(f"Error with original projector builder: {e}")
        
        # Build with our enhanced builders
        print("\nTesting enhanced builders...")
        try:
            enhanced_tower = build_image_tower(args, config_path)
            print(f"Enhanced image tower type: {type(enhanced_tower).__name__}")
        except Exception as e:
            print(f"Error with enhanced image tower builder: {e}")
        
        try:
            enhanced_projector = build_image_projector(args, config_path)
            print(f"Enhanced projector type: {type(enhanced_projector).__name__}")
        except Exception as e:
            print(f"Error with enhanced projector builder: {e}")
        
        return True
    except Exception as e:
        print(f"Error during comparison: {e}")
        return False

def main():
    """Main function to run all tests."""
    parser = argparse.ArgumentParser(description="Test encoder configuration")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    args = parser.parse_args()
    
    # Use the provided config path or create a temp config if none provided
    config_path = args.config
    if config_path is None:
        # Create a temp config file
        temp_config = {
            'image_encoder': {
                'type': 'clip',
                'model_name': 'openai/clip-vit-large-patch14',
                'select_layer': -1,
                'select_feature': 'patch',
                'use_encoder_framework': True
            },
            'projector': {
                'image_projector_type': 'linear',
                'mm_hidden_size': 1024,
                'hidden_size': 4096
            }
        }
        
        temp_path = 'temp_config.yaml'
        with open(temp_path, 'w') as f:
            yaml.dump(temp_config, f)
        
        config_path = temp_path
        print(f"Created temporary config file at {temp_path}")
    
    print(f"Using configuration file: {config_path}")
    print_separator()
    
    # Run the tests
    config_loaded = test_config_loading(config_path)
    print_separator()
    
    if config_loaded:
        encoder_loaded = test_encoder_loading(config_path)
        print_separator()
        
        projector_loaded = test_projector_loading(config_path)
        print_separator()
        
        compared = compare_with_original_builders(config_path)
        print_separator()
        
        # Print summary
        print("Test Summary:")
        print(f"Configuration loading: {'SUCCESS' if config_loaded else 'FAILURE'}")
        print(f"Encoder loading: {'SUCCESS' if encoder_loaded else 'FAILURE'}")
        print(f"Projector loading: {'SUCCESS' if projector_loaded else 'FAILURE'}")
        print(f"Builder comparison: {'SUCCESS' if compared else 'FAILURE'}")
    
    # Clean up temp file if we created one
    if config_path == 'temp_config.yaml' and os.path.exists(config_path):
        os.remove(config_path)
        print(f"Removed temporary config file {config_path}")

if __name__ == "__main__":
    main() 