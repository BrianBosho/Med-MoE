# encoder_testing.py
# This is a utility script for testing different encoders with LLaVA models

import os
import torch
import sys
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union, Any

# Make sure we can import from the moellava module
sys.path.append(os.path.abspath('.'))

class EncoderTester:
    """Utility class for testing different image encoders with LLaVA models"""
    
    def __init__(self, encoder_type="clip", vision_tower_path="openai/clip-vit-large-patch14", 
                 model_path=None, model_base=None, device="cuda"):
        """
        Initialize the encoder tester
        
        Args:
            encoder_type (str): Type of encoder to use ('clip' or 'sigclip')
            vision_tower_path (str): Path or name of the vision tower model
            model_path (str): Path to the LLaVA model
            model_base (str): Base model name/path
            device (str): Device to run on ('cuda' or 'cpu')
        """
        self.encoder_type = encoder_type
        self.vision_tower_path = vision_tower_path
        self.model_path = model_path
        self.model_base = model_base
        self.device = device
        
        # Initialize components as None
        self.vision_tower = None
        self.processor = None
        self.tokenizer = None
        self.model = None
        
        print(f"Initialized EncoderTester with {encoder_type} encoder")
    
    def load_vision_encoder(self):
        """Load the vision encoder based on the specified type"""
        print(f"Loading {self.encoder_type} vision encoder from {self.vision_tower_path}")
        
        try:
            if self.encoder_type.lower() == "clip":
                from transformers import CLIPVisionModel, CLIPImageProcessor
                
                # Create a simple vision tower that uses CLIP
                class CLIPVisionTower:
                    def __init__(self, model_path, device='cuda'):
                        self.model_path = model_path
                        self.device = device
                        self.vision_model = None
                        self.image_processor = None
                        self.is_loaded = False
                        
                    def load_model(self):
                        self.vision_model = CLIPVisionModel.from_pretrained(self.model_path)
                        self.image_processor = CLIPImageProcessor.from_pretrained(self.model_path)
                        self.vision_model.to(device=self.device, dtype=torch.float16)
                        self.is_loaded = True
                        return self.vision_model
                    
                    def encode_images(self, images):
                        if not self.is_loaded:
                            self.load_model()
                        
                        image_inputs = self.image_processor(images=images, return_tensors="pt").to(self.device)
                        with torch.no_grad():
                            outputs = self.vision_model(**image_inputs)
                        
                        return outputs.last_hidden_state
                
                self.vision_tower = CLIPVisionTower(self.vision_tower_path, self.device)
                self.vision_tower.load_model()
                self.processor = {'image': self.vision_tower.image_processor, 'video': None}
                
                print(f"Successfully loaded CLIP vision tower from {self.vision_tower_path}")
                return self.vision_tower
                
            elif self.encoder_type.lower() == "sigclip":
                from transformers import AutoModel, SigLIPImageProcessor
                
                # Create a vision tower that uses SigCLIP
                class SigCLIPVisionTower:
                    def __init__(self, model_path, device='cuda'):
                        self.model_path = model_path
                        self.device = device
                        self.vision_model = None
                        self.image_processor = None
                        self.is_loaded = False
                        
                    def load_model(self):
                        # For SigLIP, the vision model is part of the full model
                        # We'll extract the vision model part
                        self.vision_model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
                        self.vision_model = self.vision_model.vision_model  # Extract vision part
                        self.image_processor = SigLIPImageProcessor.from_pretrained(self.model_path)
                        self.vision_model.to(device=self.device, dtype=torch.float16)
                        self.is_loaded = True
                        return self.vision_model
                    
                    def encode_images(self, images):
                        if not self.is_loaded:
                            self.load_model()
                        
                        image_inputs = self.image_processor(images=images, return_tensors="pt").to(self.device)
                        with torch.no_grad():
                            outputs = self.vision_model(**image_inputs)
                        
                        return outputs.last_hidden_state
                
                self.vision_tower = SigCLIPVisionTower(self.vision_tower_path, self.device)
                self.vision_tower.load_model()
                self.processor = {'image': self.vision_tower.image_processor, 'video': None}
                
                print(f"Successfully loaded SigCLIP vision tower from {self.vision_tower_path}")
                return self.vision_tower
            
            else:
                raise ValueError(f"Unsupported encoder type: {self.encoder_type}")
                
        except Exception as e:
            print(f"Error loading vision encoder: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_model(self):
        """Load the LLaVA model using the custom vision encoder"""
        if self.model_path is None:
            print("No model path provided. Skipping model loading.")
            return None
        
        print(f"Loading LLaVA model from {self.model_path}")
        
        try:
            # Import the necessary modules
            from moellava.model.builder import load_pretrained_model
            
            # Create a custom vision config to override the default
            class CustomVisionConfig:
                def __init__(self, mm_vision_tower, encoder_type, image_size=224):
                    self.mm_vision_tower = mm_vision_tower
                    self.encoder_type = encoder_type
                    self.image_size = image_size
            
            custom_config = CustomVisionConfig(
                mm_vision_tower=self.vision_tower_path,
                encoder_type=self.encoder_type,
                image_size=224  # You can adjust this as needed
            )
            
            # Load the model with the custom configuration
            self.tokenizer, self.model, self.processor, context_len = load_pretrained_model(
                model_path=self.model_path,
                model_base=self.model_base,
                load_8bit=False,
                load_4bit=False,
                device_map=self.device,
                device=self.device,
                custom_vision_config=custom_config
            )
            
            print(f"Successfully loaded model from {self.model_path}")
            return self.model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_with_image(self, image_path, prompt="What's in this image?"):
        """Test the model with an image"""
        if self.model is None:
            print("Model not loaded. Please call load_model() first.")
            return None
        
        try:
            from PIL import Image
            import requests
            from io import BytesIO
            
            # Load image
            if image_path.startswith(('http://', 'https://')):
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                image = Image.open(image_path).convert('RGB')
            
            print(f"Processing image from {image_path}")
            
            # Prepare input for the model
            from moellava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
            
            # Check if the model config has specific tokens
            mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
            mm_use_im_patch_token = getattr(self.model.config, "mm_use_im_patch_token", True)
            
            if mm_use_im_patch_token:
                prompt = f"{DEFAULT_IMAGE_PATCH_TOKEN}\n{prompt}"
            if mm_use_im_start_end:
                prompt = f"{DEFAULT_IM_START_TOKEN}{prompt}{DEFAULT_IM_END_TOKEN}"
            
            # Process the image
            image_tensor = self.processor['image'](images=image, return_tensors="pt").to(self.device)
            
            # Tokenize the prompt
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    images=image_tensor.pixel_values,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                )
            
            # Decode the output
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Model response: {response}")
            
            return response
            
        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            return None

    def compare_encoders(self, image_path, prompt="What's in this image?", encoders=None):
        """Compare different encoders on the same image"""
        if encoders is None:
            encoders = ["clip", "sigclip"]
        
        results = {}
        
        # Save current encoder
        current_encoder = self.encoder_type
        current_vision_tower = self.vision_tower_path
        
        for encoder in encoders:
            print(f"\n--------- Testing {encoder} encoder ---------")
            self.encoder_type = encoder
            
            # Update vision tower path based on encoder if needed
            if encoder == "clip":
                self.vision_tower_path = "openai/clip-vit-large-patch14"
            elif encoder == "sigclip":
                self.vision_tower_path = "google/siglip-base-patch16-224"
            
            # Load the vision encoder and model
            self.load_vision_encoder()
            self.load_model()
            
            # Test with image
            result = self.test_with_image(image_path, prompt)
            results[encoder] = result
        
        # Restore original encoder
        self.encoder_type = current_encoder
        self.vision_tower_path = current_vision_tower
        
        return results