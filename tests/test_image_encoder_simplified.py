#!/usr/bin/env python
# Simplified diagnostic script for image encoder loading

import torch
import os
import sys
import argparse
from transformers import AutoConfig, AutoModelForCausalLM, CLIPImageProcessor, CLIPVisionModel

def test_clip_loading():
    """Test loading CLIP vision tower directly"""
    print("\n=== Testing CLIP Vision Tower Loading ===\n")
    
    try:
        # Try to load CLIP processor
        vision_tower_name = "openai/clip-vit-large-patch14"
        print(f"Loading image processor from {vision_tower_name}")
        processor = CLIPImageProcessor.from_pretrained(vision_tower_name)
        print("Successfully loaded image processor")
        
        # Try to load CLIP vision model
        print(f"Loading vision model from {vision_tower_name}")
        model = CLIPVisionModel.from_pretrained(vision_tower_name)
        print("Successfully loaded vision model")
        
        print("\nCLIP Loading Test Result: SUCCESS")
        return True
    except Exception as e:
        print(f"Error during CLIP loading: {e}")
        print("\nCLIP Loading Test Result: FAILED")
        return False

def test_model_config(model_path):
    """Test loading a model's config and check for vision tower attributes"""
    print(f"\n=== Testing Model Config from {model_path} ===\n")
    
    try:
        # Try to load the model config
        print(f"Loading config from {model_path}")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print("Successfully loaded config")
        
        # Check for vision tower attributes
        has_mm_vision_tower = hasattr(config, 'mm_vision_tower')
        mm_vision_tower_value = getattr(config, 'mm_vision_tower', None)
        print(f"Has mm_vision_tower attribute: {has_mm_vision_tower}")
        if has_mm_vision_tower:
            print(f"mm_vision_tower value: {mm_vision_tower_value}")
        
        has_mm_image_tower = hasattr(config, 'mm_image_tower')
        mm_image_tower_value = getattr(config, 'mm_image_tower', None)
        print(f"Has mm_image_tower attribute: {has_mm_image_tower}")
        if has_mm_image_tower:
            print(f"mm_image_tower value: {mm_image_tower_value}")
        
        # Try loading model head to check methods
        try:
            print("\nTrying to load model head...")
            # Load just the config to avoid full model loading
            model_type = getattr(config, 'model_type', None)
            print(f"Model type: {model_type}")
            
            # Get all config attributes as dictionary
            config_dict = config.to_dict()
            print("\nOther relevant config settings:")
            for key in config_dict:
                if 'vision' in key.lower() or 'image' in key.lower() or 'mm_' in key.lower():
                    print(f"  {key}: {config_dict[key]}")
            
            print("\nModel Config Test Result: SUCCESS")
            return True, has_mm_vision_tower, mm_vision_tower_value, has_mm_image_tower, mm_image_tower_value
        except Exception as e:
            print(f"Error loading model head: {e}")
            print("\nModel head loading failed, but config loaded successfully")
            return True, has_mm_vision_tower, mm_vision_tower_value, has_mm_image_tower, mm_image_tower_value
            
    except Exception as e:
        print(f"Error during model config loading: {e}")
        print("\nModel Config Test Result: FAILED")
        return False, False, None, False, None

def print_separator():
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test image encoder loading")
    parser.add_argument("--model_path", type=str, help="Path to the model to test", default=None)
    args = parser.parse_args()
    
    print("=== Image Encoder Loading Diagnostic ===")
    print("Python version:", sys.version)
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
    
    print_separator()
    clip_success = test_clip_loading()
    
    model_config_success = False
    has_mm_vision_tower = False
    mm_vision_tower_value = None
    has_mm_image_tower = False
    mm_image_tower_value = None
    
    if args.model_path:
        print_separator()
        model_config_success, has_mm_vision_tower, mm_vision_tower_value, has_mm_image_tower, mm_image_tower_value = test_model_config(args.model_path)
    
    print_separator()
    print("Summary of Findings:")
    if clip_success:
        print("1. CLIP vision tower and processor can be loaded directly.")
        print("   This indicates that the fallback mechanism should work correctly.")
    else:
        print("1. CLIP vision tower loading failed.")
        print("   Check your network connection and CLIP model availability.")
    
    if args.model_path:
        if model_config_success:
            print(f"\n2. Model config from {args.model_path} loaded successfully.")
            if has_mm_vision_tower:
                if mm_vision_tower_value:
                    print(f"   - mm_vision_tower is set to {mm_vision_tower_value}")
                    print("   - This should be used as the primary source for image processing")
                else:
                    print(f"   - mm_vision_tower is present but set to None")
                    print("   - This could be why the fallback is being used")
            else:
                print(f"   - mm_vision_tower attribute is missing")
                print("   - This is why the fallback is being used")
                
            if has_mm_image_tower:
                if mm_image_tower_value:
                    print(f"   - mm_image_tower is set to {mm_image_tower_value}")
                    print("   - This should be used for the second attempt at image processing")
                else:
                    print(f"   - mm_image_tower is present but set to None")
                    print("   - This is why the second attempt is failing and the fallback is being used")
            else:
                print(f"   - mm_image_tower attribute is missing")
                print("   - This is why the second attempt is failing and the fallback is being used")
        else:
            print(f"\n2. Failed to load model config from {args.model_path}.")
            print("   This may indicate issues with the model path or configuration.")
    
    print("\nRecommended fixes to avoid fallback:")
    print("1. Ensure model config has 'mm_vision_tower' attribute set to a valid path")
    print("   (e.g., 'openai/clip-vit-large-patch14')")
    print("2. Alternatively, ensure model config has 'mm_image_tower' attribute set properly")
    print("3. Make sure the model architecture implements the 'get_image_tower' method") 