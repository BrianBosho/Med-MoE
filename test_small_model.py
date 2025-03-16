#!/usr/bin/env python
"""
Test script for loading a small model with our enhanced builder.
This uses a small model to make testing faster and less resource-intensive.
"""

import os
import sys
import argparse
import torch
from moellava.model.builder_enhanced import load_pretrained_model

def parse_args():
    parser = argparse.ArgumentParser(description="Test loading a small model with enhanced builder")
    parser.add_argument("--model_path", type=str, required=False, 
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Path to the model or model name on HF Hub")
    parser.add_argument("--vision_encoder", type=str, required=False, 
                        default="openai/clip-vit-base-patch32",
                        help="Vision encoder model (using a smaller one for speed)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Testing model loading with {args.model_path} and vision encoder {args.vision_encoder}")
    
    # Create custom vision config
    class CustomVisionConfig:
        def __init__(self):
            self.mm_vision_tower = args.vision_encoder
            self.image_size = 224
            self.encoder_type = 'clip'
    
    # Load model with custom vision configuration
    try:
        tokenizer, model, processor, context_len = load_pretrained_model(
            model_path=args.model_path,
            model_base=None,
            device="cpu",  # Use CPU for testing
            custom_vision_config=CustomVisionConfig(),
            encoder_type='clip'
        )
        
        print(f"\nModel loaded successfully!")
        print(f"Tokenizer: {type(tokenizer).__name__}")
        print(f"Model: {type(model).__name__}")
        print(f"Image processor: {type(processor['image']).__name__ if processor['image'] else 'None'}")
        print(f"Context length: {context_len}")
        
        # Print model configuration details
        if hasattr(model, 'config'):
            print("\nModel Configuration:")
            if hasattr(model.config, 'mm_vision_tower'):
                print(f"  Vision tower: {model.config.mm_vision_tower}")
            if hasattr(model.config, 'mm_vision_select_layer'):
                print(f"  Vision select layer: {model.config.mm_vision_select_layer}")
        
        print("\nTest successful!")
        return True
        
    except Exception as e:
        print(f"\nError loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 