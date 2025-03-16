#!/usr/bin/env python
"""
Test script for SigClip encoder support in the enhanced builder.
This script verifies that the SigClip encoder can be loaded and used
with the enhanced model builder.
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, Any, Optional

# Ensure the right paths are set
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import the enhanced builder
from moellava.model.builder_enhanced import (
    load_pretrained_model,
    load_vision_encoder
)

# Test configuration
@dataclass
class TestConfig:
    mm_vision_tower: str = "google/siglip-base-patch16-224"
    image_size: int = 224
    vision_feature_layer: int = -2
    vision_feature_select_strategy: str = "default"

def test_sigclip_encoder_loading():
    """Test loading the SigClip encoder directly"""
    print("\n=== Testing SigClip Encoder Loading ===")
    
    config = TestConfig()
    
    # Test with explicit encoder type
    print("\nTesting with explicit 'sigclip' encoder type:")
    encoder, processor = load_vision_encoder(
        config, 
        encoder_type='sigclip',
        custom_vision_config=config
    )
    
    if encoder:
        print(f"✓ Successfully loaded custom SigClip encoder: {type(encoder).__name__}")
    else:
        print("✗ Failed to load custom SigClip encoder")
    
    if processor:
        print(f"✓ Successfully loaded image processor: {type(processor).__name__}")
    else:
        print("✗ Failed to load image processor")
    
    # Test with automatic encoder type detection
    print("\nTesting with automatic encoder type detection:")
    encoder, processor = load_vision_encoder(
        config,
        custom_vision_config=config
    )
    
    if encoder:
        print(f"✓ Successfully loaded auto-detected SigClip encoder: {type(encoder).__name__}")
    else:
        print("✗ Failed to load auto-detected SigClip encoder")
    
    if processor:
        print(f"✓ Successfully loaded image processor: {type(processor).__name__}")
    else:
        print("✗ Failed to load image processor")
    
    return encoder, processor

def test_model_with_sigclip():
    """Test loading a small model with SigClip encoder"""
    print("\n=== Testing Full Model with SigClip Encoder ===")
    
    # Configure a small model with SigClip
    model_config = {
        "model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "mm_vision_tower": "google/siglip-base-patch16-224",
        "mm_hidden_size": 768
    }
    
    try:
        # Load the model with SigClip encoder
        model, image_processor, tokenizer, context_len = load_pretrained_model(
            model_config,
            encoder_type='sigclip',
        )
        
        print(f"\n✓ Successfully loaded model with SigClip encoder")
        print(f"  - Model type: {type(model).__name__}")
        print(f"  - Image processor type: {type(image_processor).__name__}")
        print(f"  - Tokenizer type: {type(tokenizer).__name__}")
        print(f"  - Context length: {context_len}")
        
        return model, image_processor, tokenizer
    except Exception as e:
        print(f"\n✗ Error loading model with SigClip encoder: {e}")
        return None, None, None

if __name__ == "__main__":
    # Test loading just the encoder
    encoder, processor = test_sigclip_encoder_loading()
    
    # Test loading a full model with the SigClip encoder
    model, image_processor, tokenizer = test_model_with_sigclip() 