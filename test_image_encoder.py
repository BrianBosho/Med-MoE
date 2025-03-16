#!/usr/bin/env python
# Diagnostic script to check why image encoder falls back to default

import sys
import os
import torch
from moellava.model.builder import load_pretrained_model

def print_separator():
    print("\n" + "="*80 + "\n")

def inspect_model_config(model_path, model_base=None):
    """Inspects the model configuration to debug image encoder loading issues"""
    print(f"Testing model: {model_path}")
    print(f"Model base: {model_base}")
    
    try:
        # Load the model with verbose output
        print_separator()
        print("ATTEMPTING TO LOAD MODEL...")
        tokenizer, model, processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=model_base,
            model_name=os.path.basename(model_path),
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        print_separator()
        print("MODEL LOADED SUCCESSFULLY")
        
        # Check processor
        print("\nProcessor state:")
        for key, value in processor.items():
            print(f"  - {key}: {'Loaded successfully' if value is not None else 'NOT LOADED'}")
        
        # Check config attributes
        print("\nModel configuration attributes:")
        if hasattr(model, 'config'):
            print(f"  - Has 'mm_vision_tower' attribute: {hasattr(model.config, 'mm_vision_tower')}")
            if hasattr(model.config, 'mm_vision_tower'):
                print(f"  - mm_vision_tower value: {model.config.mm_vision_tower}")
            
            print(f"  - Has 'mm_image_tower' attribute: {hasattr(model.config, 'mm_image_tower')}")
            if hasattr(model.config, 'mm_image_tower'):
                print(f"  - mm_image_tower value: {model.config.mm_image_tower}")
        
        # Check model methods
        print("\nModel methods:")
        print(f"  - Has 'get_image_tower' method: {hasattr(model, 'get_image_tower')}")
        
        if hasattr(model, 'get_image_tower'):
            try:
                image_tower = model.get_image_tower()
                print(f"  - Image tower: {'Loaded' if image_tower is not None else 'None'}")
                print(f"  - Is tower loaded: {image_tower.is_loaded if hasattr(image_tower, 'is_loaded') else 'Unknown'}")
            except Exception as e:
                print(f"  - Error getting image tower: {e}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_image_encoder.py <model_path> [model_base]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    model_base = sys.argv[2] if len(sys.argv) > 2 else None
    
    inspect_model_config(model_path, model_base) 