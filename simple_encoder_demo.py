#!/usr/bin/env python
"""
Simple demo script that shows how to select different encoders with MoELLaVA.
"""

import argparse
from PIL import Image
from encoders.selector import load_model_with_encoder, ENCODER_TYPES
import os

def parse_args():
    parser = argparse.ArgumentParser(description="MoELLaVA Encoder Selection Demo")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the MoELLaVA model")
    
    # Image input
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the image file to process")
    
    # Encoder selection - THIS IS THE KEY PARAMETER
    parser.add_argument("--encoder", type=str, default="clip", choices=list(ENCODER_TYPES.keys()),
                        help=f"Encoder type to use: {', '.join(ENCODER_TYPES.keys())}")
    
    # Optional encoder model path
    parser.add_argument("--encoder_path", type=str, default=None,
                        help="Optional custom path to the encoder model")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 50)
    print("MoELLaVA Encoder Selection Demo")
    print("=" * 50)
    print(f"Loading model from: {args.model_path}")
    print(f"Using encoder: {args.encoder}")
    if args.encoder_path:
        print(f"Using custom encoder path: {args.encoder_path}")
    print("-" * 50)
    
    # Extract model name from path if not already a plain name
    model_name = os.path.basename(args.model_path.rstrip('/'))
    print(f"Model name: {model_name}")
    
    # Load the model with the specified encoder - THIS IS THE KEY PART
    tokenizer, model, processor, context_len = load_model_with_encoder(
        model_path=args.model_path,
        encoder=args.encoder,  # Choose: "clip", "sigclip", "medclip", etc.
        encoder_path=args.encoder_path,  # Optional: specify model path
        model_name=model_name  # Pass the extracted model name
    )
    
    # Load the image
    image = Image.open(args.image).convert('RGB')
    print(f"Loaded image: {args.image}")
    
    # Get the image embeddings
    image_embeddings = model.encode_images(image)
    print(f"Generated image embeddings with shape: {image_embeddings.shape}")
    
    # Generate a caption
    prompt = "Describe this medical image in detail."
    
    # Generate and print the caption
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Prepare multimodal inputs with the image
    mm_inputs = model.prepare_inputs_labels_for_multimodal(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        position_ids=None,
        past_key_values=None,
        labels=None,
        images=image
    )
    
    # Generate a response
    print("\nGenerating caption...")
    generated_ids = model.generate(
        **mm_inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    
    # Decode the response
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("\nModel Response:")
    print("-" * 50)
    print(response)
    print("-" * 50)
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main() 