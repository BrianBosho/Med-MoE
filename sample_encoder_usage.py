#!/usr/bin/env python
"""
Sample script demonstrating how to use the encoder selection system.

This script shows how to:
1. Load a MoELLaVA model with different vision encoders
2. Switch between encoders at runtime
3. Compare results from different encoders
"""

import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import json

from encoders.selector import load_model_with_encoder, get_available_encoders
from moellava.conversation import conv_templates


def parse_args():
    parser = argparse.ArgumentParser(description="MoELLaVA Encoder Selection Demo")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the MoELLaVA model")
    parser.add_argument("--model_base", type=str, default=None,
                        help="Optional base model path")
    
    # Encoder selection arguments
    parser.add_argument("--encoder", type=str, default="clip",
                        help=f"Encoder type to use. Available: {', '.join(get_available_encoders())}")
    parser.add_argument("--encoder_path", type=str, default=None,
                        help="Optional custom path to the encoder model")
    
    # Compare mode
    parser.add_argument("--compare", action="store_true",
                        help="Compare multiple encoders on the same image")
    parser.add_argument("--encoders_to_compare", type=str, nargs="+", 
                        default=["clip", "sigclip", "blip", "dino"],
                        help="List of encoders to compare")
    
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


def visualize_embeddings(image_embeddings, title, output_path="embeddings_viz.png"):
    """Create a simple visualization of image embeddings"""
    plt.figure(figsize=(10, 6))
    
    # Take first 100 dimensions if embeddings are too large
    if image_embeddings.shape[1] > 100:
        display_embeddings = image_embeddings[:, :100]
    else:
        display_embeddings = image_embeddings
    
    plt.imshow(display_embeddings.cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title(f"{title} (shape: {image_embeddings.shape})")
    plt.xlabel("Embedding Dimensions")
    plt.ylabel("Patches or Tokens")
    plt.savefig(output_path)
    print(f"Embedding visualization saved to: {output_path}")
    return output_path


def compare_encoders(args):
    """Compare multiple encoders on the same image"""
    print(f"Comparing encoders: {args.encoders_to_compare}")
    
    # Load the image
    image = Image.open(args.image).convert('RGB')
    
    results = {}
    for encoder_name in args.encoders_to_compare:
        print(f"\nLoading model with {encoder_name} encoder...")
        try:
            # Load model with the current encoder
            tokenizer, model, processor, context_len = load_model_with_encoder(
                model_path=args.model_path,
                model_base=args.model_base,
                encoder=encoder_name,
                encoder_path=args.encoder_path,
                load_8bit=args.load_8bit,
                load_4bit=args.load_4bit,
                device=args.device
            )
            
            # Process the image
            with torch.no_grad():
                # Get image embeddings
                image_tensor = processor["image"](image, return_tensors="pt").pixel_values.to(args.device)
                image_embeds = model.get_vision_tower()(image_tensor)
                
                # Generate a caption
                prompt = "Describe this image in detail."
                conv = conv_templates["llava_v1"].copy()
                conv.append_message(conv.roles[0], f"{prompt}")
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                
                input_ids = tokenizer([prompt], return_tensors="pt").input_ids.to(args.device)
                stop_str = conv.sep if conv.sep_style != conv_templates["llava_v1"].sep_style else None
                
                # Prepare inputs for generation
                input_ids = model.prepare_inputs_for_generation(
                    input_ids, images=image_tensor, 
                    return_dict=True
                )
                
                # Generate text
                output_ids = model.generate(
                    **input_ids,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
                
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                try:
                    response = outputs.split(prompt)[1].strip()
                except:
                    response = outputs
                
                # Visualize the embeddings
                viz_path = visualize_embeddings(
                    image_embeds, 
                    f"{encoder_name.upper()} Embeddings", 
                    f"embeddings_{encoder_name}.png"
                )
                
                # Save results
                results[encoder_name] = {
                    "embedding_shape": image_embeds.shape,
                    "embedding_viz_path": viz_path,
                    "response": response,
                }
                
                print(f"Generated response with {encoder_name}:")
                print("-" * 40)
                print(response)
                print("-" * 40)
            
        except Exception as e:
            print(f"Error with {encoder_name} encoder: {e}")
            results[encoder_name] = {"error": str(e)}
    
    # Save comparison results
    with open("encoder_comparison.json", "w") as f:
        json.dump({k: v for k, v in results.items() if "error" not in v}, f, indent=2)
    
    print("\nComparison completed! Results saved to encoder_comparison.json")
    return results


def main():
    args = parse_args()
    
    if args.compare:
        # Compare multiple encoders
        compare_encoders(args)
    else:
        # Use a single encoder
        print(f"Loading MoELLaVA model from: {args.model_path}")
        print(f"Using encoder: {args.encoder}")
        
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
        
        # Process the image
        with torch.no_grad():
            # Get image processor
            image_processor = processor['image']
            
            # Process the image and get embeddings
            image_tensor = image_processor(image, return_tensors="pt").pixel_values.to(args.device)
            image_embeds = model.get_vision_tower()(image_tensor)
            
            # Print embedding information
            print(f"Image embedding shape: {image_embeds.shape}")
            
            # Visualize the embeddings
            visualize_embeddings(image_embeds, f"{args.encoder.upper()} Embeddings")
            
            # Generate a caption
            prompt = "Describe this image in detail."
            conv = conv_templates["llava_v1"].copy()
            conv.append_message(conv.roles[0], f"{prompt}")
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            input_ids = tokenizer([prompt], return_tensors="pt").input_ids.to(args.device)
            stop_str = conv.sep if conv.sep_style != conv_templates["llava_v1"].sep_style else None
            
            # Prepare inputs for generation
            input_ids = model.prepare_inputs_for_generation(
                input_ids, images=image_tensor, 
                return_dict=True
            )
            
            # Generate text
            output_ids = model.generate(
                **input_ids,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            try:
                response = outputs.split(prompt)[1].strip()
            except:
                response = outputs
            
            print("\nGenerated Caption:")
            print("-" * 40)
            print(response)
            print("-" * 40)
        
        print("\nDemo completed successfully!")


if __name__ == "__main__":
    main() 