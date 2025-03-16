#!/usr/bin/env python
"""
Demo script for encoder selection in MoELLaVA.

This script demonstrates how to use the encoder selection interface to 
easily choose different vision encoders for MoELLaVA.
"""

import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt
import os

from encoders.selector import load_model_with_encoder, ENCODER_TYPES


def parse_args():
    parser = argparse.ArgumentParser(description="MoELLaVA Encoder Selection Demo")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the MoELLaVA model")
    parser.add_argument("--model_base", type=str, default=None,
                        help="Optional base model path")
    
    # Encoder selection arguments
    parser.add_argument("--encoder", type=str, default="clip", choices=list(ENCODER_TYPES.keys()),
                        help=f"Encoder type to use: {', '.join(ENCODER_TYPES.keys())}")
    parser.add_argument("--encoder_path", type=str, default=None,
                        help="Optional custom path to the encoder model")
    
    # Image input
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the image file to process")
    
    # Model loading options
    parser.add_argument("--load_8bit", action="store_true",
                        help="Load model in 8-bit precision")
    parser.add_argument("--load_4bit", action="store_true",
                        help="Load model in 4-bit precision")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to load model on (cuda, cpu)")
    
    return parser.parse_args()


def visualize_embeddings(image_embeddings, output_path="embeddings_viz.png"):
    """Create a simple visualization of image embeddings"""
    plt.figure(figsize=(10, 6))
    
    # Take first 100 dimensions if embeddings are too large
    if image_embeddings.shape[1] > 100:
        image_embeddings = image_embeddings[:, :100]
    
    plt.imshow(image_embeddings.cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title(f"Image Embedding Visualization (shape: {image_embeddings.shape})")
    plt.xlabel("Embedding Dimensions")
    plt.ylabel("Patches")
    plt.savefig(output_path)
    print(f"Embedding visualization saved to: {output_path}")


def main():
    args = parse_args()
    
    print(f"Loading MoELLaVA model from: {args.model_path}")
    print(f"Using encoder: {args.encoder}")
    if args.encoder_path:
        print(f"Using custom encoder path: {args.encoder_path}")
    
    # Load the model with the specified encoder
    tokenizer, model, processor, context_len = load_model_with_encoder(
        model_path=args.model_path,
        model_base=args.model_base,
        encoder=args.encoder,
        encoder_path=args.encoder_path,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device=args.device
    )
    
    # Load and process image
    print(f"Processing image: {args.image}")
    image = Image.open(args.image).convert('RGB')
    
    # Get image processor
    image_processor = processor['image']
    
    # Process the image
    with torch.no_grad():
        # Encode the image using the model's built-in methods
        image_embeddings = model.encode_images(image)
    
    # Print embedding information
    print(f"Image embedding shape: {image_embeddings.shape}")
    
    # Visualize the embeddings
    visualize_embeddings(image_embeddings)
    
    # Generate a simple caption using the model
    if hasattr(model, 'generate'):
        prompt = "Describe this image in detail."
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
        
        # Replace with image token handling as per MoELLaVA's protocol
        # This is a simplified example and may need to be adapted
        image_token_id = tokenizer.get_vocab().get('<image>', None)
        if image_token_id:
            inputs = model.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                images=image,
                position_ids=None,
                attention_mask=None,
                past_key_values=None,
                labels=None
            )
            
            # Generate the caption
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
            # Decode the outputs
            caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("\nGenerated Caption:")
            print(caption)
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main() 