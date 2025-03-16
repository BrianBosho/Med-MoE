#!/usr/bin/env python
"""
Test script for the enhanced model builder.
This script tests the configuration loading and vision encoder customization.
"""

import os
import sys
import argparse
import torch
from moellava.model.builder_enhanced import (
    load_model_config, 
    load_vision_encoder, 
    load_tokenizer,
    load_pretrained_model
)

def parse_args():
    parser = argparse.ArgumentParser(description="Test the enhanced model builder")
    parser.add_argument("--model_path", type=str, required=False, 
                        default="liuhaotian/llava-v1.5-7b",
                        help="Path to the model or model name on HF Hub")
    parser.add_argument("--vision_encoder", type=str, required=False, 
                        default="openai/clip-vit-large-patch14",
                        help="Vision encoder model name")
    parser.add_argument("--test_mode", type=str, required=False, 
                        choices=["config", "encoder", "full"],
                        default="config",
                        help="Test mode: config, encoder, or full model loading")
    return parser.parse_args()

def test_config_loading(model_path, vision_encoder):
    """Test loading and customizing the model configuration"""
    print(f"Testing config loading for {model_path} with vision encoder {vision_encoder}")
    
    # Test default config loading
    default_config = load_model_config(model_path)
    print("\nDefault config:")
    print(f"  Vision tower: {getattr(default_config, 'mm_vision_tower', 'None')}")
    
    # Test custom vision config
    vision_config = {
        'model_name': vision_encoder,
        'select_layer': -2,
        'select_feature': 'patch',
        'image_size': 224
    }
    
    custom_config = load_model_config(model_path, vision_config=vision_config)
    print("\nCustom config:")
    print(f"  Vision tower: {getattr(custom_config, 'mm_vision_tower', 'None')}")
    print(f"  Vision select layer: {getattr(custom_config, 'mm_vision_select_layer', 'None')}")
    print(f"  Vision select feature: {getattr(custom_config, 'mm_vision_select_feature', 'None')}")
    print(f"  Vision image size: {getattr(custom_config, 'mm_vision_image_size', 'None')}")
    
    return custom_config

def test_vision_encoder(config):
    """Test loading the vision encoder"""
    print("\nTesting vision encoder loading")
    
    # Test CLIP encoder
    print("Loading default CLIP encoder...")
    vision_encoder, image_processor = load_vision_encoder(config)
    print(f"Image processor: {type(image_processor).__name__}")
    
    # Test with custom encoder type if available
    try:
        print("\nTrying custom encoder loading (this might fail if encoders package is not available)...")
        vision_encoder, image_processor = load_vision_encoder(
            config, 
            encoder_type="clip"
        )
        print(f"Custom encoder loaded: {vision_encoder is not None}")
        print(f"Image processor: {type(image_processor).__name__}")
    except Exception as e:
        print(f"Custom encoder loading failed (expected): {e}")
    
    return image_processor

def test_full_model(model_path, vision_encoder):
    """Test loading the full model with custom vision encoder"""
    print(f"\nTesting full model loading with custom vision encoder: {vision_encoder}")
    
    # Create custom vision config
    class CustomVisionConfig:
        def __init__(self):
            self.mm_vision_tower = vision_encoder
            self.image_size = 224
    
    try:
        # Load model with custom vision config
        tokenizer, model, processor, context_len = load_pretrained_model(
            model_path=model_path,
            custom_vision_config=CustomVisionConfig(),
            device="cpu"  # Use CPU for testing
        )
        
        print(f"Model loaded successfully")
        print(f"Tokenizer: {type(tokenizer).__name__}")
        print(f"Model: {type(model).__name__}")
        print(f"Image processor: {type(processor['image']).__name__ if processor['image'] else 'None'}")
        print(f"Context length: {context_len}")
        
        # Check if vision tower path is correctly set
        if hasattr(model.config, 'mm_vision_tower'):
            print(f"Model vision tower: {model.config.mm_vision_tower}")
        
        return True
    except Exception as e:
        print(f"Full model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    args = parse_args()
    
    if args.test_mode == "config" or args.test_mode == "full":
        config = test_config_loading(args.model_path, args.vision_encoder)
    
    if args.test_mode == "encoder" or args.test_mode == "full":
        if 'config' not in locals():
            config = load_model_config(args.model_path, vision_config={'model_name': args.vision_encoder})
        image_processor = test_vision_encoder(config)
    
    if args.test_mode == "full":
        success = test_full_model(args.model_path, args.vision_encoder)
        if success:
            print("\nAll tests passed successfully!")
        else:
            print("\nSome tests failed. Check the logs above for details.")
            sys.exit(1)
    
    print("\nTests completed.") 