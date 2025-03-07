# Image Encoders Framework

This package provides a modular framework for using different image encoders in your projects. Currently supported encoders:

- **CLIP**: OpenAI's Contrastive Language-Image Pre-training model
- **SigCLIP**: Google's Sigmoid Loss Pre-training model (from Transformers)

## Installation

First, install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from encoders import create_encoder
from PIL import Image

# Create an encoder (default is CLIP)
encoder = create_encoder("clip")

# Or specify the model variant
encoder = create_encoder("clip", model_name="ViT-L/14")

# Load an image
image = Image.open("path/to/image.jpg")

# Encode the image
embedding = encoder.encode(image)

# Print embedding dimension
print(f"Embedding dimension: {encoder.get_embedding_dim()}")
```

### Switching Encoders

Switching between encoders is as simple as changing the encoder type:

```python
# Use CLIP
clip_encoder = create_encoder("clip")

# Use SigCLIP
sigclip_encoder = create_encoder("sigclip")
```

### Processing Multiple Images

You can encode multiple images at once:

```python
images = [Image.open(f) for f in ["image1.jpg", "image2.jpg", "image3.jpg"]]
embeddings = encoder.encode(images)
```

## Adding New Encoders

To add a new encoder, create a new file in the `encoders` directory that inherits from `ImageEncoder`:

1. Create a new file (e.g., `encoders/my_encoder.py`)
2. Implement the required methods (`encode`, `get_embedding_dim`) 
3. Update the factory in `encoders/factory.py`
4. Import the new encoder in `encoders/__init__.py`

Example template:

```python
from .base import ImageEncoder

class MyEncoder(ImageEncoder):
    def __init__(self, model_name="default/model", device=None):
        super().__init__(device=device)
        # Initialize your model here
        
    def encode(self, images):
        # Implement encoding logic
        pass
        
    def get_embedding_dim(self):
        # Return embedding dimension
        pass
```

## Command-Line Example

The repository includes an example script that demonstrates how to use the encoders:

```bash
python encoder_example.py --encoder clip
python encoder_example.py --encoder sigclip
``` 