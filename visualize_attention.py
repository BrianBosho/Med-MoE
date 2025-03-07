import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from moellava.conversation import conv_templates, SeparatorStyle
from moellava.model.builder import load_pretrained_model
from moellava.utils import disable_torch_init
from moellava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def visualize_attention(model, tokenizer, image_processor, image, prompt, device="cuda"):
    # Process image
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [img.to(device, dtype=torch.float16) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(device, dtype=torch.float16)
    
    # Prepare conversation
    if model.config.mm_use_im_start_end:
        prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
    else:
        prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    
    # Tokenize input
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)
    
    # Run model with output_attentions=True to get attention weights
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=512,
            use_cache=True,
            output_attentions=True,  # This is key to get attention weights
            return_dict_in_generate=True,  # Return a dict with all outputs
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )
    
    # Extract generated text
    generated_ids = outputs.sequences[0, input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Extract attention weights
    # The structure of attention weights depends on the model architecture
    # This is a general approach that might need adjustments for specific models
    attention_weights = outputs.attentions
    
    return generated_text, attention_weights, image


def plot_attention_maps(image, attention_weights, output_path=None):
    """
    Plot attention maps overlaid on the image.
    
    Args:
        image: PIL Image
        attention_weights: Attention weights from the model
        output_path: Path to save the visualization
    """
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Process attention weights
    # This will depend on the specific structure of your model's attention weights
    # Here's a general approach that might need adjustments
    
    # For example, if attention_weights is a tuple of tensors for each layer:
    # We'll take the last layer's attention and average across heads
    if isinstance(attention_weights, tuple):
        # Take the last layer's attention
        last_layer_attn = attention_weights[-1]
        
        # Average across attention heads
        if len(last_layer_attn.shape) == 4:  # [batch, heads, seq_len, seq_len]
            attn_map = last_layer_attn[0].mean(dim=0).cpu().numpy()
        else:
            # Handle other shapes as needed
            attn_map = last_layer_attn[0].cpu().numpy()
    else:
        # Handle other attention weight structures
        attn_map = attention_weights
    
    # Find the attention to the image tokens
    # This depends on your tokenization scheme
    # Assuming image tokens are at the beginning of the sequence
    # and there are 576 image tokens (typical for 24x24 patches)
    image_attn = attn_map[:, :576]  # Adjust based on your model
    
    # Reshape to match image dimensions (assuming 24x24 patches)
    # You'll need to adjust this based on your model's patch size
    height, width = 24, 24
    attn_map_reshaped = image_attn.reshape(attn_map.shape[0], height, width)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original image
    axes[0].imshow(img_array)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Plot attention map overlaid on image
    axes[1].imshow(img_array)
    # Use a colormap for the attention heatmap
    attn_heatmap = axes[1].imshow(attn_map_reshaped.mean(axis=0), alpha=0.5, cmap="hot")
    axes[1].set_title("Attention Map")
    axes[1].axis("off")
    
    # Add colorbar
    plt.colorbar(attn_heatmap, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, 
        args.load_8bit, args.load_4bit, device=args.device
    )
    
    image_processor = processor['image']
    
    # Load image
    image = load_image(args.image_file)
    
    # Run inference with attention visualization
    generated_text, attention_weights, image = visualize_attention(
        model, tokenizer, image_processor, image, args.prompt, args.device
    )
    
    print(f"Generated text: {generated_text}")
    
    # Plot attention maps
    plot_attention_maps(image, attention_weights, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()
    main(args)
