#!/usr/bin/env python
# Notebook-friendly code for testing image encoder loading

import torch
import os
import sys
import logging
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

# Add MedVQA path to system path if needed
sys.path.append('/home/brian_bosho/MedVQA/Med-MoE')  # Update this path as needed

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Cell 1: Load the model ===
# Run this cell to load the model

def load_model(model_path, model_name=None, use_4bit=False):
    """Load the model and return components"""
    from moellava.model.builder import load_pretrained_model
    
    # Default model name if not provided
    if model_name is None:
        model_name = os.path.basename(model_path)
    
    # Configure device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set model loading parameters
    kwargs = {}
    if use_4bit:
        kwargs["load_4bit"] = True
        print("Using 4-bit quantization")
    
    # Load the model
    print(f"Loading model from {model_path}")
    try:
        tokenizer, model, processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            device=device,
            **kwargs
        )
        print("✅ Model loaded successfully")
        
        # Check the processor
        if processor is None:
            processor = {'image': None, 'video': None}
        
        print(f"Image processor: {'Available' if processor['image'] is not None else 'Not available'}")
        
        return tokenizer, model, processor, context_len
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

# Example usage:
# model_path = "MedMoE-phi2"  # Update this path
# tokenizer, model, processor, context_len = load_model(model_path, model_name="phi2", use_4bit=True)


# === Cell 2: Process an image ===
# Run this cell after loading the model to process an image

def process_image(processor, model, image_path):
    """Process an image with the model's processor or fallback methods"""
    # Load image
    image = Image.open(image_path).convert("RGB")
    print(f"Input image shape: {image.size}")
    
    # Display image
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title("Input Image")
    plt.show()
    
    # Get device
    device = model.device if hasattr(model, 'device') else "cuda" if torch.cuda.is_available() else "cpu"
    
    # Handle processor not available case
    if processor is None or 'image' not in processor or processor['image'] is None:
        print("⚠️ No image processor found. Attempting to use fallback methods.")
        
        # Try to load CLIP processor as fallback
        try:
            from transformers import CLIPImageProcessor
            print("Loading CLIP processor as fallback")
            clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
            
            # Update processor dict
            if processor is None:
                processor = {'image': clip_processor}
            else:
                processor['image'] = clip_processor
            
            # Process with CLIP
            image_tensor = processor['image'](images=image, return_tensors='pt')['pixel_values']
            print(f"✅ Processed with fallback CLIP processor: {image_tensor.shape}")
        except Exception as e:
            print(f"❌ CLIP fallback failed: {e}")
            
            # Ultimate fallback - basic preprocessing
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)
            print(f"✅ Processed with basic preprocessing: {image_tensor.shape}")
    else:
        # Try to use the model's processor
        try:
            # Different processor methods
            if hasattr(processor['image'], '__call__'):
                image_tensor = processor['image'](images=image, return_tensors='pt')['pixel_values']
                print("Using processor's __call__ method")
            elif hasattr(processor['image'], 'preprocess'):
                image_tensor = processor['image'].preprocess(image, return_tensors='pt')['pixel_values']
                print("Using processor's preprocess method")
            else:
                raise AttributeError("Image processor has no usable method")
                
            print(f"✅ Processed with model's processor: {image_tensor.shape}")
        except Exception as e:
            print(f"❌ Error with model processor: {e}")
            
            # Fallback to basic preprocessing
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)
            print(f"✅ Processed with basic preprocessing: {image_tensor.shape}")
    
    # Move to device
    image_tensor = image_tensor.to(device, dtype=torch.float16)
    return image_tensor, processor

# Example usage:
# image_path = "/home/brian_bosho/MedVQA/Med-MoE/image.jpg"  # Update this path
# image_tensor, processor = process_image(processor, model, image_path)


# === Cell 3: Generate a response ===
# Run this cell after processing the image to generate a response

def generate_response(tokenizer, model, processor, image_tensor, prompt="Describe this image in detail."):
    """Generate a response from the model based on the image and prompt"""
    from moellava.mm_utils import tokenizer_image_token
    from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from moellava.conversation import conv_templates
    
    # Make sure we have all components
    if tokenizer is None or model is None or image_tensor is None:
        print("❌ Missing required components")
        return None
    
    # Prepare the prompt with image token
    if hasattr(model.config, "mm_use_im_start_end") and model.config.mm_use_im_start_end:
        formatted_prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
    else:
        formatted_prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    
    print(f"Prompt: {prompt}")
    
    # Set up conversation
    conv_mode = "phi" if "phi" in model.__class__.__name__.lower() else "llava_v0"
    print(f"Using conversation mode: {conv_mode}")
    
    try:
        conv = conv_templates[conv_mode].copy()
    except KeyError:
        print(f"⚠️ Conversation mode {conv_mode} not found, trying default")
        conv = conv_templates["llava_v0"].copy()
    
    # Add the prompt to the conversation
    conv.append_message(conv.roles[0], formatted_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # Tokenize
    device = model.device
    try:
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
        
        # Generate response
        print("Generating response...")
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=512,
                use_cache=True
            )
            
        # Decode the response
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        # Extract the response part only (after separator)
        if hasattr(conv, 'sep2') and conv.sep2 in outputs:
            response = outputs.split(conv.sep2)[-1].strip()
        elif hasattr(conv, 'sep') and conv.sep in outputs:
            response = outputs.split(conv.sep)[-1].strip()
        else:
            response = outputs.strip()
            
        print("\n=== Model Response ===")
        print(response)
        return response
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        import traceback
        traceback.print_exc()
        return None

# Example usage:
# response = generate_response(tokenizer, model, processor, image_tensor, prompt="Describe what you see in this medical image.") 