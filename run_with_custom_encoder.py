#!/usr/bin/env python
"""
Example script that demonstrates how to run Med-MoE with custom image encoders
and projectors using our enhanced configuration approach.
"""
import os
import sys
import argparse
import torch
from PIL import Image

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our configuration tools
try:
    from encoders import load_config, config_to_args
    ENCODERS_FRAMEWORK_AVAILABLE = True
except ImportError:
    ENCODERS_FRAMEWORK_AVAILABLE = False
    print("WARNING: Encoders framework not available.")
    sys.exit(1)

# Import from moellava
from moellava.model.builder import load_pretrained_model
from moellava.conversation import conv_templates, SeparatorStyle
from moellava.mm_utils import tokenizer_image_token
from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

def print_separator():
    print("\n" + "="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Run Med-MoE with custom encoder configuration")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model weights")
    parser.add_argument("--model-base", type=str, default=None, help="Path to the base model (optional)")
    parser.add_argument("--config", type=str, required=True, help="Path to encoder configuration YAML file")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.", help="Text prompt for the model")
    parser.add_argument("--conv-mode", type=str, default="llava_v1", help="Conversation mode")
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file {args.config} does not exist.")
        sys.exit(1)
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist.")
        sys.exit(1)
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    print(f"Using image encoder: {config.image_encoder.type} - {config.image_encoder.model_name}")
    print(f"Using projector type: {config.projector.image_projector_type}")
    
    # Load model with our configuration
    print("\nLoading model...")
    tokenizer, model, processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name=os.path.basename(args.model_path),
        config_path=args.config  # Pass our config path
    )
    
    # Check if we have an image processor
    if not processor['image']:
        print("Error: Image processor is not available.")
        sys.exit(1)
    
    print(f"Successfully loaded model and image processor.")
    
    # Process an image if provided
    if args.image and os.path.exists(args.image):
        print(f"\nProcessing image: {args.image}")
        image = Image.open(args.image).convert('RGB')
        
        # Prepare input prompt
        if getattr(model.config, 'mm_use_im_start_end', False):
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + args.prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + args.prompt
        
        # Setup conversation template
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Process the image into pixel values
        image_tensor = processor['image'](image, return_tensors='pt')['pixel_values'].to(model.device, dtype=torch.float16)
        
        # Tokenize the input
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        
        # Generate response
        print("\nGenerating response...")
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=512,
                use_cache=True
            )
        
        # Decode and print output
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        answer = outputs.split(conv.sep2)[-1].strip()
        print_separator()
        print("Response:")
        print(answer)
        print_separator()
    else:
        print("\nNo image provided or image file does not exist. Skipping inference.")
    
    print("Done.")

if __name__ == "__main__":
    main() 