import torch
from PIL import Image
from .base import ImageEncoder

class ClipEncoder(ImageEncoder):
    """CLIP-based image encoder using HuggingFace Transformers."""
    
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        """
        Initialize a CLIP encoder.
        
        Args:
            model_name: CLIP model variant to use 
                       (e.g., 'openai/clip-vit-base-patch32', 'openai/clip-vit-large-patch14')
            device: Device to run the model on ('cpu', 'cuda', etc.)
        """
        super().__init__(device=device)
        
        # Import here to not require the dependency if not using this encoder
        try:
            from transformers import CLIPProcessor, CLIPVisionModel
        except ImportError:
            raise ImportError(
                "Transformers is not installed. Please install it with: pip install transformers"
            )
        
        # Load the model and processor
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Determine embedding dimension based on the model
        self._embedding_dim = self.model.config.hidden_size
        
    def encode(self, images):
        """
        Encode images using CLIP.
        
        Args:
            images: List of PIL images or tensor
            
        Returns:
            Image embeddings as tensor
        """
        with torch.no_grad():
            # Handle different input types
            if isinstance(images, torch.Tensor):
                # This is more complex for CLIP since it expects PIL images
                raise ValueError("CLIP encoder expects PIL images, not tensors")
            elif isinstance(images, list):
                # Process list of PIL images
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            elif isinstance(images, Image.Image):
                # Process a single PIL image
                inputs = self.processor(images=[images], return_tensors="pt").to(self.device)
            else:
                raise ValueError(f"Unsupported image type: {type(images)}")
            
            # Generate embeddings
            outputs = self.model(**inputs)
            
            # Use pooler output (corresponds to [CLS] token representation)
            embeddings = outputs.pooler_output
            
            # Normalize embeddings
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
            return embeddings
    
    def get_embedding_dim(self):
        """Return embedding dimension."""
        return self._embedding_dim 