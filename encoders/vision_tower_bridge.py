import torch
import torch.nn as nn
from .factory import create_encoder

class EncoderVisionTower(nn.Module):
    """
    Bridge class that adapts our encoder framework to the vision tower interface
    expected by the Med-MoE codebase.
    """
    
    def __init__(self, image_tower, args, delay_load=False, cache_dir='./cache_dir'):
        super().__init__()
        
        self.is_loaded = False
        self.image_tower_name = image_tower
        self.cache_dir = cache_dir
        
        # Parse options from args
        self.select_layer = getattr(args, 'mm_vision_select_layer', -1)
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        
        # Get image size if provided in args
        self.image_size = getattr(args, 'image_size', None)
        
        # Override encoder_type if specified in args
        self.encoder_type = getattr(args, 'encoder_type', None)
        
        # If encoder_type not explicitly provided, determine from image tower name
        if self.encoder_type is None:
            if 'clip' in image_tower.lower():
                self.encoder_type = 'clip'
            elif 'siglip' in image_tower.lower() or 'sigclip' in image_tower.lower():
                self.encoder_type = 'sigclip'
            else:
                # Default to CLIP
                self.encoder_type = 'clip'
        
        # Print info about the tower configuration
        print(f"EncoderVisionTower: {self.image_tower_name}")
        print(f"  Encoder type: {self.encoder_type}")
        print(f"  Feature selection: {self.select_feature}")
        if self.image_size:
            print(f"  Image size: {self.image_size}")
        
        if not delay_load:
            self.load_model()
    
    def load_model(self):
        """Load the encoder model."""
        # Pass image_size if we have it
        encoder_kwargs = {}
        if self.image_size:
            encoder_kwargs['image_size'] = self.image_size
            
        self.encoder = create_encoder(
            encoder_type=self.encoder_type,
            model_name=self.image_tower_name,
            device=self._get_device(),
            **encoder_kwargs
        )
        self.is_loaded = True
    
    def _get_device(self):
        """Get the device to use."""
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def feature_select(self, image_features):
        """
        Select features based on configuration.
        
        Note: This simplified implementation assumes the encoder 
        already provides the right format. For more complex cases,
        you might need to adapt this.
        """
        if self.select_feature == 'patch':
            # Skip the [CLS] token if present, otherwise return all features
            if image_features.shape[1] > 1:
                return image_features[:, 1:]
            return image_features
        elif self.select_feature == 'cls_patch':
            # Keep all tokens including [CLS]
            return image_features
        elif self.select_feature == 'cls':
            # Return only the [CLS] token if present, otherwise first token
            return image_features[:, 0:1]
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
    
    @torch.no_grad()
    def forward(self, images):
        """
        Process images through the encoder.
        
        Args:
            images: Either a list of image tensors or a batch tensor
            
        Returns:
            Encoded image features
        """
        if not self.is_loaded:
            self.load_model()
        
        if isinstance(images, list):
            # Process a list of images
            result = self.encoder.encode(images)
            # Apply feature selection
            return self.feature_select(result)
        else:
            # Process a batch of images
            result = self.encoder.encode(images)
            # Apply feature selection
            return self.feature_select(result)
    
    @property
    def dummy_feature(self):
        """Return a dummy feature for initialization."""
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)
    
    @property
    def dtype(self):
        """Return the data type of the encoder."""
        if hasattr(self, 'encoder') and self.is_loaded:
            return next(self.encoder.model.parameters()).dtype
        return torch.float32
    
    @property
    def device(self):
        """Return the device of the encoder."""
        if hasattr(self, 'encoder') and self.is_loaded:
            return next(self.encoder.model.parameters()).device
        return self._get_device()
    
    @property
    def hidden_size(self):
        """Return the hidden size of the encoder."""
        if hasattr(self, 'encoder') and self.is_loaded:
            return self.encoder.get_embedding_dim()
        return 768  # Default value 