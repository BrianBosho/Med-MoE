#!/usr/bin/env python
# Test script for loading models with image encoders

import torch
import os
import logging
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_safely(model_path, model_name=None, use_4bit=False):
    """
    Load a model safely by handling potential errors
    """
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from moellava.model.builder import load_pretrained_model
        
        # Default to base model name if not provided
        if model_name is None:
            model_name = os.path.basename(model_path)
        
        # Configure device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Skip loading MoE models which might have compatibility issues
        kwargs = {}
        if use_4bit:
            kwargs["load_4bit"] = True
        
        # Load the model with safer parameters
        logger.info(f"Loading model from {model_path}")
        tokenizer, model, processor, context_len = load_pretrained_model(
            model_path=model_path, 
            model_base=None, 
            model_name=model_name,
            device=device,
            **kwargs
        )
        
        logger.info(f"Successfully loaded model")
        return tokenizer, model, processor, context_len
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def process_image(processor, model, image_path):
    """
    Process an image using the model's processor or a fallback method
    """
    # Load image from path
    image = Image.open(image_path).convert("RGB")
    logger.info(f"Input image shape: {image.size}")
    
    try:
        # Display image if in a notebook
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis("off")
        plt.show()
    except:
        pass
    
    device = model.device if hasattr(model, 'device') else "cuda" if torch.cuda.is_available() else "cpu"
    
    # Process the image - with proper error handling
    if processor is None or 'image' not in processor or processor['image'] is None:
        logger.warning("No image processor found. Using fallback processing.")
        
        # Attempt to load CLIP processor as fallback
        try:
            from transformers import CLIPImageProcessor
            logger.info("Loading CLIP processor as fallback")
            clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
            
            # Update the processor dict
            if processor is None:
                processor = {'image': clip_processor}
            else:
                processor['image'] = clip_processor
            
            # Process with CLIP processor
            image_tensor = processor['image'](images=image, return_tensors='pt')['pixel_values']
            logger.info(f"Processed with fallback CLIP processor: {image_tensor.shape}")
        except Exception as clip_e:
            logger.error(f"Failed to use CLIP fallback: {clip_e}")
            # Ultimate fallback - basic preprocessing
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)
            logger.info(f"Processed with basic fallback: {image_tensor.shape}")
    else:
        try:
            # Try with the processor's direct call method
            if hasattr(processor['image'], '__call__'):
                image_tensor = processor['image'](images=image, return_tensors='pt')['pixel_values']
            # Try with preprocess method
            elif hasattr(processor['image'], 'preprocess'):
                image_tensor = processor['image'].preprocess(image, return_tensors='pt')['pixel_values']
            else:
                raise AttributeError("Image processor has no callable method or preprocess method")
                
            logger.info(f"Processed image with model processor: {image_tensor.shape}")
        except Exception as proc_e:
            logger.error(f"Error using model processor: {proc_e}")
            # Fallback to basic preprocessing
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)
            logger.info(f"Processed with basic fallback: {image_tensor.shape}")
    
    # Move to device
    image_tensor = image_tensor.to(device, dtype=torch.float16)
    return image_tensor, processor

def test_model_with_image(model_path, image_path, model_name=None, use_4bit=False):
    """
    Test loading a model and processing an image
    """
    # Load model
    tokenizer, model, processor, context_len = load_model_safely(model_path, model_name, use_4bit)
    
    if model is None:
        logger.error("Model loading failed")
        return
    
    # Process image
    image_tensor, processor = process_image(processor, model, image_path)
    
    logger.info("Successfully processed image for model")
    return tokenizer, model, processor, image_tensor

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test model loading and image processing")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--model_name", type=str, default=None, help="Model name")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")
    
    args = parser.parse_args()
    test_model_with_image(args.model, args.image, args.model_name, args.use_4bit) 