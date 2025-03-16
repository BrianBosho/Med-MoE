#!/usr/bin/env python
"""
Simple test script for SigClip encoder.
"""

import argparse
from PIL import Image
from encoders.factory import create_encoder

def parse_args():
    parser = argparse.ArgumentParser(description="Test SigClip encoder")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--model", type=str, default="google/siglip-base-patch16-224", 
                        help="Model name or path")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Testing with image: {args.image}")
    print(f"Using model: {args.model}")
    
    # Load the image
    image = Image.open(args.image).convert('RGB')
    
    # Create the SigClip encoder
    encoder = create_encoder(
        encoder_type="sigclip",
        model_name=args.model
    )
    
    # Encode the image
    embedding = encoder.encode(image)
    
    # Print embedding information
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding dimension: {encoder.get_embedding_dim()}")
    
    # Sample a few values
    print(f"Sample values: {embedding[0, :5]}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main() 