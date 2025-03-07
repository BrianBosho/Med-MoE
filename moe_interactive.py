import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import BytesIO

from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from moellava.conversation import conv_templates, SeparatorStyle
from moellava.model.builder import load_pretrained_model
from moellava.utils import disable_torch_init
from moellava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

class MedVQAInteractive:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.image = None
        self.image_tensor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conv_mode = None
        self.roles = None
        self.conv = None
        
    def setup_model(self, model_path, model_base=None, load_8bit=False, load_4bit=False):
        """
        Load the model, tokenizer, and processor
        """
        disable_torch_init()
        
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, processor, context_len = load_pretrained_model(
            model_path, model_base, model_name, load_8bit, load_4bit, device=self.device
        )
        
        self.image_processor = processor['image']
        
        # Determine conversation mode
        if 'qwen' in model_name.lower():
            self.conv_mode = "qwen"
        elif 'openchat' in model_name.lower():
            self.conv_mode = "openchat"
        elif 'phi' in model_name.lower():
            self.conv_mode = "phi"
        elif 'stablelm' in model_name.lower():
            self.conv_mode = "stablelm"
        else:
            if 'llama-2' in model_name.lower():
                self.conv_mode = "llava_llama_2"
            elif "v1" in model_name.lower():
                self.conv_mode = "llava_v1"
            elif "mpt" in model_name.lower():
                self.conv_mode = "mpt"
            else:
                self.conv_mode = "llava_v0"
        
        self.conv = conv_templates[self.conv_mode].copy()
        
        if "mpt" in model_name.lower():
            self.roles = ('user', 'assistant')
        else:
            self.roles = self.conv.roles
            
        print(f"Model loaded successfully. Conversation mode: {self.conv_mode}")
        return self
    
    def load_image(self, image_path):
        """
        Load and process the image
        """
        # Load image
        if image_path.startswith('http://') or image_path.startswith('https://'):
            response = requests.get(image_path)
            self.image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            self.image = Image.open(image_path).convert('RGB')
        
        # Process image
        self.image_tensor = process_images([self.image], self.image_processor, self.model.config)
        if type(self.image_tensor) is list:
            self.image_tensor = [img.to(self.device, dtype=torch.float16) for img in self.image_tensor]
        else:
            self.image_tensor = self.image_tensor.to(self.device, dtype=torch.float16)
        
        print(f"Image loaded and processed successfully")
        return self
    
    def ask(self, question, temperature=0.2, max_new_tokens=512, get_attention=True):
        """
        Ask a question about the loaded image and get the response with attention maps
        """
        if self.model is None or self.image_tensor is None:
            raise ValueError("Model or image not loaded. Call setup_model() and load_image() first.")
        
        # Reset conversation for a new question
        self.conv = conv_templates[self.conv_mode].copy()
        
        # Format prompt with image token
        if self.model.config.mm_use_im_start_end:
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + question
        
        # Add to conversation
        self.conv.append_message(self.conv.roles[0], prompt)
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()
        
        # Tokenize
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        attention_mask = torch.ones_like(input_ids)
        
        # Set up stopping criteria
        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        
        attention_weights = None
        
        # Generate response
        with torch.inference_mode():
            if get_attention:
                try:
                    # Try to get attention weights
                    outputs = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        images=self.image_tensor,
                        do_sample=True if temperature > 0 else False,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        use_cache=True,
                        output_attentions=True,
                        return_dict_in_generate=True,
                        stopping_criteria=[stopping_criteria],
                        pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                    )
                    
                    # Extract generated text
                    generated_ids = outputs.sequences[0, input_ids.shape[1]:]
                    response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    
                    # Get attention weights
                    attention_weights = outputs.attentions
                    
                except Exception as e:
                    print(f"Error getting attention weights: {e}")
                    print("Falling back to standard generation without attention weights")
                    get_attention = False
            
            if not get_attention:
                # Standard generation without attention weights
                output_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    images=self.image_tensor,
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                )
                
                # Extract generated text
                generated_ids = output_ids[0, input_ids.shape[1]:]
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Update conversation with response
        self.conv.messages[-1][-1] = response
        
        return response, attention_weights
    
    def visualize_attention(self, attention_weights, output_path=None):
        """
        Visualize attention maps from the model's response
        """
        if attention_weights is None:
            print("No attention weights available to visualize")
            return
        
        if self.image is None:
            print("No image loaded to visualize attention on")
            return
        
        # Convert image to numpy array
        img_array = np.array(self.image)
        
        # Process attention weights based on their structure
        try:
            # For models where attention_weights is a tuple of tuples
            if isinstance(attention_weights, tuple):
                # Take the last layer's attention
                last_layer_attn = attention_weights[-1]
                
                # If last_layer_attn is also a tuple
                if isinstance(last_layer_attn, tuple) and len(last_layer_attn) > 0:
                    # Usually the first element contains the attention weights
                    attn_tensor = last_layer_attn[0]
                    
                    # Convert to numpy
                    if hasattr(attn_tensor, 'cpu'):
                        attn_map = attn_tensor.cpu().numpy()
                    else:
                        print("Cannot process attention weights: unexpected structure")
                        return
                else:
                    print("Last layer attention has unexpected structure")
                    return
            else:
                print("Unexpected attention weights type")
                return
            
            # Process the attention map for visualization
            # This depends on your model's architecture and might need adjustment
            
            # For a typical attention map with shape [batch, heads, seq_len, seq_len]
            if len(attn_map.shape) == 4:
                # Average across attention heads
                attn_map = attn_map.mean(axis=1)[0]  # Take first batch, average across heads
                
                # Find the attention to the image tokens
                # Assuming image tokens are at the beginning and there are image_token_count tokens
                image_token_count = 576  # Adjust based on your model (e.g., 24x24 patches)
                
                # Check if we have enough tokens
                if attn_map.shape[1] >= image_token_count:
                    # Extract attention to image tokens
                    image_attn = attn_map[:, :image_token_count]
                    
                    # Reshape to match image dimensions
                    height, width = 24, 24  # Adjust based on your model's patch size
                    try:
                        attn_map_reshaped = image_attn.reshape(-1, height, width).mean(axis=0)
                    except ValueError:
                        print(f"Cannot reshape attention map of shape {image_attn.shape} to ({height}, {width})")
                        # Use a flattened version instead
                        attn_map_reshaped = image_attn.mean(axis=0).reshape(height, width, order='F')
                else:
                    print(f"Not enough tokens in attention map (needed {image_token_count}, got {attn_map.shape[1]})")
                    # Use whatever tokens we have
                    attn_map_reshaped = attn_map.mean(axis=0).reshape(24, 24, order='F')
            else:
                print(f"Unexpected attention map shape: {attn_map.shape}")
                # Try to use a flattened version
                attn_map_reshaped = attn_map.mean(axis=0) if len(attn_map.shape) > 2 else attn_map
                # Reshape to a square for visualization
                side_len = int(np.sqrt(attn_map_reshaped.size))
                attn_map_reshaped = attn_map_reshaped.flatten()[:side_len**2].reshape(side_len, side_len)
            
            # Create a figure with subplots
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Plot original image
            axes[0].imshow(img_array)
            axes[0].set_title("Original Image")
            axes[0].axis("off")
            
            # Plot attention map overlaid on image
            axes[1].imshow(img_array)
            # Use a colormap for the attention heatmap
            attn_heatmap = axes[1].imshow(attn_map_reshaped, alpha=0.5, cmap="hot")
            axes[1].set_title("Attention Map")
            axes[1].axis("off")
            
            # Add colorbar
            plt.colorbar(attn_heatmap, ax=axes[1], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path)
                print(f"Saved attention map to {output_path}")
            
            return fig
            
        except Exception as e:
            print(f"Error visualizing attention: {e}")
            import traceback
            traceback.print_exc()
            return None
