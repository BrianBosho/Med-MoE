import torch
from PIL import Image
from .base import ImageEncoder

class SigClipEncoder(ImageEncoder):
    """SigCLIP-based image encoder."""
    
    def __init__(self, model_name="google/siglip-base-patch16-224", device=None):
        """
        Initialize a SigCLIP encoder.
        
        Args:
            model_name: SigCLIP model variant to use 
                       (e.g., 'google/siglip-base-patch16-224', 'google/siglip-large-patch16-384')
            device: Device to run the model on ('cpu', 'cuda', etc.)
        """
        super().__init__(device=device)
        
        # Import here to not require the dependency if not using this encoder
        try:
            from transformers import AutoProcessor, AutoModel
        except ImportError:
            raise ImportError(
                "Transformers is not installed. Please install it with: pip install transformers"
            )
        
        # Load the processor and model
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Load the full model first
        full_model = AutoModel.from_pretrained(model_name)
        
        # Use only the vision model component since we only need image embeddings
        self.model = full_model.vision_model.to(self.device)
        
        # Extract the hidden_size from the vision_config
        self._embedding_dim = full_model.config.vision_config.hidden_size
        
    def encode(self, images):
        """
        Encode images using SigCLIP.
        
        Args:
            images: List of PIL images or tensor
            
        Returns:
            Image embeddings as tensor
        """
        with torch.no_grad():
            # Handle different input types
            if isinstance(images, torch.Tensor):
                # This is more complex for SigLip since it expects PIL images
                # For simplicity, we'll raise an error
                raise ValueError("SigCLIP encoder expects PIL images, not tensors")
            elif isinstance(images, list):
                # Process list of PIL images
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            elif isinstance(images, Image.Image):
                # Process a single PIL image
                inputs = self.processor(images=[images], return_tensors="pt").to(self.device)
            else:
                raise ValueError(f"Unsupported image type: {type(images)}")
            
            # Extract just the pixel_values for the vision model
            pixel_values = inputs.pixel_values if hasattr(inputs, 'pixel_values') else inputs['pixel_values']
            
            # Generate embeddings using only the vision model component
            outputs = self.model(pixel_values=pixel_values)
            
            # SigLip vision model uses the pooler_output for image embeddings
            embeddings = outputs.pooler_output
            
            # Normalize embeddings (important for comparison)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
            return embeddings
    
    def get_embedding_dim(self):
        """Return embedding dimension."""
        return self._embedding_dim 