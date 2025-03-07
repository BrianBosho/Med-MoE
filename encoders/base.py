from abc import ABC, abstractmethod
import torch

class ImageEncoder(ABC):
    """Abstract base class for all image encoders."""
    
    def __init__(self, device=None):
        """
        Initialize the encoder.
        
        Args:
            device: Device to run the model on ('cpu', 'cuda', etc.)
                   If None, will use CUDA if available, else CPU.
        """
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def encode(self, images):
        """
        Encode a batch of images into embeddings.
        
        Args:
            images: List of PIL images or tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Tensor of shape [batch_size, embedding_dim]
        """
        pass
    
    @abstractmethod
    def get_embedding_dim(self):
        """Return the dimension of the embeddings."""
        pass
        
    @property
    def device(self):
        """Return the device where the model is located."""
        return self._device
    
    def __call__(self, images):
        """Allow encoders to be called directly."""
        return self.encode(images) 