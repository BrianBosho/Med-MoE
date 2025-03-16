"""
BiomedCLIP encoder implementation for using Microsoft's Biomedical CLIP models.
"""

import torch
import numpy as np
from PIL import Image
import open_clip

class BiomedClipEncoder:
    """BiomedCLIP-based image encoder using OpenCLIP."""
    
    def __init__(self, model_name_or_path, config=None):
        """
        Initialize a BiomedCLIP encoder.
        
        Args:
            model_name_or_path: Model identifier or path
                (e.g., "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
            config: Configuration object with optional parameters
        """
        self.model_name_or_path = model_name_or_path
        self.config = config
        self.is_loaded = False
        self.model = None
        self.image_processor = None
        
        # Ensure model path is correctly formatted for open_clip
        if not self.model_name_or_path.startswith('hf-hub:'):
            # Handle the case where the model path might be given in different formats
            if 'biomedclip' in self.model_name_or_path.lower() or 'medclip' in self.model_name_or_path.lower():
                # Extract the model name from the path
                model_name = self.model_name_or_path.split('/')[-1] if '/' in self.model_name_or_path else self.model_name_or_path
                self.model_name_or_path = f"hf-hub:microsoft/{model_name}"
            else:
                # Default to a known BiomedCLIP model if path doesn't seem to be one
                print(f"Warning: '{self.model_name_or_path}' doesn't appear to be a BiomedCLIP model path.")
                print("Defaulting to 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'")
                self.model_name_or_path = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        
        print(f"Initializing BiomedCLIP with model: {self.model_name_or_path}")
        
    def load_model(self):
        """Load the BiomedCLIP model and processor."""
        try:
            # Create model from pretrained
            self.model, self.image_processor = open_clip.create_model_from_pretrained(self.model_name_or_path)
            
            # Set model to eval mode
            self.model.eval()
            self.is_loaded = True
            print(f"Successfully loaded BiomedCLIP model: {self.model_name_or_path}")
        except Exception as e:
            print(f"Error loading BiomedCLIP model: {e}")
            raise
        
    def to(self, **kwargs):
        """Move model to specified device and dtype."""
        if self.model is not None:
            self.model = self.model.to(**kwargs)
        return self
    
    def preprocess_image(self, image):
        """
        Preprocess a single image using the model's processor.
        
        Args:
            image: PIL Image or tensor
            
        Returns:
            Preprocessed image tensor
        """
        if not self.is_loaded:
            self.load_model()
            
        # Handle different image input types
        if isinstance(image, torch.Tensor):
            # Assume it's already preprocessed properly
            return image
        elif isinstance(image, Image.Image):
            # Preprocess PIL image
            return self.image_processor(image).unsqueeze(0)
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL image then preprocess
            return self.image_processor(Image.fromarray(image)).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
    def forward(self, images):
        """
        Encode images using BiomedCLIP.
        
        Args:
            images: Tensor of preprocessed images
            
        Returns:
            Tensor of image embeddings
        """
        if not self.is_loaded:
            self.load_model()
            
        # Process images - assume images are already preprocessed
        with torch.no_grad():
            # BiomedCLIP returns image features from its forward method
            # We need to extract just the image features without text
            image_features = self.model.encode_image(images)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Reshape to match expected format (batch_size, sequence_length, hidden_size)
            # For pooled features, we create a sequence of one token
            hidden_size = image_features.size(-1)
            batch_size = images.size(0)
            
            # Check if we should return pooled or token features based on config
            select_feature = getattr(self.config, 'mm_vision_select_feature', 'patch') if self.config else 'patch'
            
            if select_feature == 'patch':
                # For patch features, we need to get the intermediate hidden states
                # However, OpenCLIP doesn't expose these by default
                # As a workaround, we'll create patches ourselves
                patch_size = 16  # Most CLIP models use 16x16 patches
                img_size = images.size(-1)
                num_patches = (img_size // patch_size) ** 2
                
                # Create artificial patch embeddings from the pooled features
                # This is a fallback since BiomedCLIP might not expose patch embeddings
                patch_embeddings = image_features.unsqueeze(1).expand(-1, num_patches, -1)
                return patch_embeddings
            else:
                # Return pooled features (CLS token equivalent)
                return image_features.unsqueeze(1) 