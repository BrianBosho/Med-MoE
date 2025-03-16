"""
SigClip encoder implementation for using Google's SigLip models.
"""

import torch
from transformers import AutoModel, AutoImageProcessor

class SigClipEncoder:
    """SigCLIP-based image encoder."""
    
    def __init__(self, model_name_or_path, config=None):
        """
        Initialize a SigCLIP encoder.
        
        Args:
            model_name_or_path: SigCLIP model variant to use 
                (e.g., "google/siglip-base-patch16-224")
            config: Configuration object with optional parameters
        """
        self.model_name_or_path = model_name_or_path
        self.config = config
        self.is_loaded = False
        self.model = None
        self.image_processor = None
        
    def load_model(self):
        """Load the SigLip vision model and image processor."""
        self.model = AutoModel.from_pretrained(self.model_name_or_path)
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_name_or_path)
        self.is_loaded = True
        
    def to(self, **kwargs):
        """Move model to specified device and dtype."""
        if self.model is not None:
            self.model = self.model.to(**kwargs)
        return self
        
    def forward(self, images):
        """
        Encode images using SigCLIP.
        
        Args:
            images: Tensor of preprocessed images
            
        Returns:
            Tensor of image embeddings
        """
        if not self.is_loaded:
            self.load_model()
            
        # Check if images are already preprocessed tensors
        if isinstance(images, torch.Tensor):
            outputs = self.model(pixel_values=images, output_hidden_states=True)
        else:
            # For PIL images, preprocess first
            raise ValueError("SigCLIP encoder expects preprocessed tensors, not PIL images")
        
        # Get the hidden states from the selected layer
        select_layer = getattr(self.config, 'mm_vision_select_layer', -2)
        hidden_states = outputs.hidden_states[select_layer]
        
        # Return the appropriate feature type
        select_feature = getattr(self.config, 'mm_vision_select_feature', 'patch')
        if select_feature == 'patch':
            return hidden_states[:, 1:]  # Skip the CLS token
        else:
            return hidden_states[:, 0]  # Return only the CLS token 