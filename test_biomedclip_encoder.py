#!/usr/bin/env python
"""
Test script for BiomedCLIP encoder support in the enhanced builder.
This script verifies that the BiomedCLIP encoder can be loaded and used
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
    mm_vision_tower: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    image_size: int = 224
    vision_feature_layer: int = -2
    vision_feature_select_strategy: str = "default"

def test_biomedclip_encoder_loading():
    """Test loading the BiomedCLIP encoder directly"""
    print("\n=== Testing BiomedCLIP Encoder Loading ===")
    
    config = TestConfig()
    
    # Test with explicit encoder type
    print("\nTesting with explicit 'biomedclip' encoder type:")
    encoder, processor = load_vision_encoder(
        config, 
        encoder_type='biomedclip',
        custom_vision_config=config
    )
    
    if encoder:
        print(f"✓ Successfully loaded custom BiomedCLIP encoder: {type(encoder).__name__}")
    else:
        print("✗ Failed to load custom BiomedCLIP encoder")
    
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
        print(f"✓ Successfully loaded auto-detected BiomedCLIP encoder: {type(encoder).__name__}")
    else:
        print("✗ Failed to load auto-detected BiomedCLIP encoder")
    
    if processor:
        print(f"✓ Successfully loaded image processor: {type(processor).__name__}")
    else:
        print("✗ Failed to load image processor")
    
    return encoder, processor

def test_model_with_biomedclip():
    """Test loading a small model with BiomedCLIP encoder"""
    print("\n=== Testing Full Model with BiomedCLIP Encoder ===")
    
    # Configure a small model with BiomedCLIP
    model_config = {
        "model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "mm_vision_tower": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "mm_hidden_size": 768
    }
    
    try:
        # Load the model with BiomedCLIP encoder
        model, image_processor, tokenizer, context_len = load_pretrained_model(
            model_config,
            encoder_type='biomedclip',
        )
        
        print(f"\n✓ Successfully loaded model with BiomedCLIP encoder")
        print(f"  - Model type: {type(model).__name__}")
        print(f"  - Image processor type: {type(image_processor).__name__}")
        print(f"  - Tokenizer type: {type(tokenizer).__name__}")
        print(f"  - Context length: {context_len}")
        
        return model, image_processor, tokenizer
    except Exception as e:
        print(f"\n✗ Error loading model with BiomedCLIP encoder: {e}")
        return None, None, None

if __name__ == "__main__":
    # Test loading just the encoder
    encoder, processor = test_biomedclip_encoder_loading()
    
    # Test loading a full model with the BiomedCLIP encoder
    model, image_processor, tokenizer = test_model_with_biomedclip() 