import torch
from PIL import Image
import argparse
from encoders import create_encoder
import requests
from io import BytesIO
import os

def download_image(url):
    """Download an image from a URL and return a PIL Image."""
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

def load_image(path_or_url):
    """Load an image from a file path or URL."""
    if path_or_url.startswith("http"):
        return download_image(path_or_url)
    else:
        return Image.open(path_or_url).convert("RGB")

def main():
    parser = argparse.ArgumentParser(description='Encode images using different models')
    parser.add_argument('--encoder', type=str, default='clip', choices=['clip', 'sigclip'],
                        help='Which encoder to use')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Specific model variant to use')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run on (cuda, cpu)')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to an image file or URL')
    args = parser.parse_args()
    
    # Define model name based on encoder type if not specified
    if args.model_name is None:
        if args.encoder == 'clip':
            args.model_name = 'openai/clip-vit-large-patch14'
        elif args.encoder == 'sigclip':
            args.model_name = 'google/siglip-base-patch16-224'
    
    # Create the encoder
    print(f"Creating {args.encoder} encoder with model {args.model_name}...")
    encoder = create_encoder(
        encoder_type=args.encoder,
        model_name=args.model_name,
        device=args.device
    )
    
    # Get images to encode
    if args.image:
        # Use provided image
        print(f"Loading image from {args.image}...")
        images = [load_image(args.image)]
    else:
        # Download sample images
        print("Downloading sample images...")
        image_urls = [
            "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
            "https://raw.githubusercontent.com/pytorch/hub/master/images/cat.jpg",
        ]
        images = [download_image(url) for url in image_urls]
    
    # Encode images
    print("Encoding images...")
    with torch.no_grad():
        embeddings = encoder.encode(images)
    
    # Print embedding information
    print(f"Encoder: {args.encoder}")
    print(f"Model: {args.model_name}")
    print(f"Embedding dimension: {encoder.get_embedding_dim()}")
    print(f"Generated {len(embeddings)} embeddings of shape {embeddings.shape}")
    
    # Calculate similarity (cosine similarity)
    if len(embeddings) > 1:
        similarity = torch.nn.functional.cosine_similarity(
            embeddings[0].unsqueeze(0),
            embeddings[1].unsqueeze(0)
        )
        print(f"Similarity between first and second image: {similarity.item():.4f}")

if __name__ == "__main__":
    main() 