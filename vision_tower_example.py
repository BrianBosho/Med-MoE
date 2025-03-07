import torch
import argparse
from PIL import Image
import os
from encoders import EncoderVisionTower
from types import SimpleNamespace

def main():
    parser = argparse.ArgumentParser(description='Test the Vision Tower Bridge')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to an image file')
    parser.add_argument('--model', type=str, default='openai/clip-vit-large-patch14',
                        help='Model name to use')
    parser.add_argument('--encoder_type', type=str, choices=['clip', 'sigclip'],
                        default=None, help='Explicitly set encoder type')
    args = parser.parse_args()
    
    # Create a configuration object similar to what the model would use
    config = SimpleNamespace(
        mm_vision_select_layer=-1,
        mm_vision_select_feature='patch'
    )
    
    # Create the vision tower
    print(f"Creating vision tower with model {args.model}...")
    vision_tower = EncoderVisionTower(args.model, config)
    
    # If encoder_type is specified, override the automatic detection
    if args.encoder_type:
        vision_tower.encoder_type = args.encoder_type
        vision_tower.load_model()
    
    # Load an image
    if args.image and os.path.exists(args.image):
        print(f"Loading image from {args.image}...")
        image = Image.open(args.image).convert('RGB')
    else:
        # Create a dummy image if none provided
        print("Creating a dummy image...")
        image = Image.new('RGB', (224, 224), color='red')
    
    # Process the image (first convert to tensor)
    image_tensor = torch.from_numpy(
        # Simple conversion to tensor - in a real app, use proper preprocessing
        torch.tensor(image).numpy().astype('float32') / 255.0
    ).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
    
    # Process through vision tower
    print("Processing image through vision tower...")
    with torch.no_grad():
        features = vision_tower(image_tensor)
    
    # Print information about the output
    print(f"Output features shape: {features.shape}")
    print(f"Using encoder type: {vision_tower.encoder_type}")
    print(f"Feature selection method: {vision_tower.select_feature}")
    
    # Show how this would integrate with the existing workflow
    print("\nExample integration in existing code:")
    print("--------------------------------------")
    print("from encoders import EncoderVisionTower")
    print("from moellava.model.multimodal_encoder.builder import build_vision_tower")
    print("\n# Modify the builder.py to include this option:")
    print("def build_vision_tower(image_tower, args):")
    print("    if 'use_encoder_framework' in args and args.use_encoder_framework:")
    print("        return EncoderVisionTower(image_tower, args)")
    print("    elif 'clip-vit' in image_tower.lower():")
    print("        return CLIPVisionTower(image_tower, args)")
    print("    elif 'siglip' in image_tower.lower():")
    print("        return SigLipVisionTower(image_tower, args)")
    print("    else:")
    print("        raise ValueError(f'Unknown image tower: {image_tower}')")

if __name__ == "__main__":
    main() 