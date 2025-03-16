"""
Basic encoder module for demonstration purposes.
In a real implementation, this would contain more sophisticated vision encoders.
"""

import torch
from transformers import CLIPVisionModel

class EncoderVisionTower:
    def __init__(self, model_name_or_path, config=None):
        self.model_name_or_path = model_name_or_path
        self.config = config
        self.is_loaded = False
        self.model = None
        
    def load_model(self):
        """Load the vision model."""
        self.model = CLIPVisionModel.from_pretrained(self.model_name_or_path)
        self.is_loaded = True
        
    def to(self, **kwargs):
        """Move model to specified device and dtype."""
        if self.model is not None:
            self.model = self.model.to(**kwargs)
        return self
        
    def forward(self, images):
        """Process images through the vision model."""
        if not self.is_loaded:
            self.load_model()
            
        outputs = self.model(images, output_hidden_states=True)
        
        # Get the hidden states from the selected layer
        select_layer = getattr(self.config, 'mm_vision_select_layer', -2)
        hidden_states = outputs.hidden_states[select_layer]
        
        # Return the appropriate feature type
        select_feature = getattr(self.config, 'mm_vision_select_feature', 'patch')
        if select_feature == 'patch':
            return hidden_states[:, 1:]  # Skip the CLS token
        else:
            return hidden_states[:, 0]  # Return only the CLS token

# Import specific encoder implementations
from .sigclip import SigClipEncoder
from .biomedclip import BiomedClipEncoder

# Import factory later to avoid circular imports
# from .factory import create_encoder

__all__ = ['EncoderVisionTower', 'SigClipEncoder', 'BiomedClipEncoder'] 