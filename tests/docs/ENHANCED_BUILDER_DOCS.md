# MoELLaVA Enhanced Builder Documentation

This comprehensive guide explains the enhanced model builder for MoELLaVA, which provides a modular and flexible architecture for loading multimodal models with various vision encoders and projectors.

## Table of Contents

- [Overview](#overview)
- [Vision Encoders](#vision-encoders)
- [Multimodal Projectors](#multimodal-projectors)
- [Configuration Options](#configuration-options)
- [Usage Examples](#usage-examples)
- [Implementation Details](#implementation-details)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Extensions](#extensions)
- [References](#references)

## Overview

The enhanced builder module provides a significant improvement over the original MoELLaVA builder:

- **Modular Architecture**: Clear separation of components for easier maintenance and extension
- **Flexible Configuration**: Support for object-based and dictionary-based configurations
- **Pluggable Components**: Easily swap vision encoders and projectors for different use cases
- **Better Error Handling**: Meaningful error messages and graceful fallbacks
- **Comprehensive Model Support**: Works with LLaMA, Mistral, Phi, Qwen, MiniCPM, StableLM, and their MoE variants

## Vision Encoders

Vision encoders process images into feature representations that can be used by multimodal models. The enhanced builder supports several encoder types:

| Encoder Type | Description | Model Examples | Strengths |
|--------------|-------------|----------------|-----------|
| **CLIP** | OpenAI's Contrastive Language-Image Pre-training model (default) | `openai/clip-vit-large-patch14` | General purpose, robust to many visual domains |
| **SigClip** | Google's SigLip model using sigmoid loss functions | `google/siglip-base-patch16-224` | Better zero-shot classification, improved alignment |
| **BiomedCLIP** | Microsoft's biomedical CLIP model | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | Specialized for medical imagery and biomedical applications |

### CLIP Encoder

CLIP (Contrastive Language-Image Pre-training) is the default encoder used in most LLaVA variants. It provides robust visual feature extraction pre-trained on a diverse set of image-text pairs.

**Key characteristics:**
- Uses Vision Transformer (ViT) architecture
- Extracts features at different resolutions (patch sizes)
- Works well for general visual domains

### SigClip Encoder

SigClip (based on Google's SigLip model) uses sigmoid loss functions instead of contrastive loss, resulting in better alignment between image and text embeddings.

**Key characteristics:**
- Improved performance on zero-shot classification tasks
- Better handling of fine-grained distinctions
- More stable training dynamics

**Requirements:**
- Transformers library (version 4.34.0 or higher)

### BiomedCLIP Encoder

BiomedCLIP is Microsoft's vision-language model specifically trained on biomedical data, making it ideal for medical applications.

**Key characteristics:**
- Specialized for medical imagery
- Trained on biomedical literature and images
- Better understanding of medical concepts

**Requirements:**
- `open_clip_torch` library (version 2.23.0 or higher)
- Transformers library (version 4.35.2 or higher recommended)

```bash
pip install open_clip_torch==2.23.0 transformers==4.35.2
```

## Multimodal Projectors

Projectors serve as the bridge between vision encoders and language models, transforming visual features into a representation compatible with the language model's embedding space.

| Projector Type | Description | Best Used With | Parameters |
|----------------|-------------|----------------|------------|
| **MLP** | Multi-layer perceptron with activation (default) | General purpose | `hidden_size`, `num_hidden_layers` |
| **Identity** | Simple pass-through without transformation | When dimensions already match | None |
| **QFormer** | Query-based transformer projector (like in BLIP-2) | Complex alignment tasks | `num_query_tokens`, `qformer_hidden_size` |
| **Perceiver** | Dynamic resampling of visual features | Fine-grained visual understanding | `num_latents`, `latent_dim` |

### MLP Projector

The default projector in most LLaVA models, consisting of one or more linear layers with activation functions in between.

**Key characteristics:**
- Simple and effective for most use cases
- Configurable number of hidden layers
- Low computational overhead

### Identity Projector

A simple pass-through projector that doesn't transform the visual features.

**Key characteristics:**
- No transformation of features
- Useful when visual features are already compatible
- Zero computational overhead

### QFormer Projector

Based on the BLIP-2 architecture, the QFormer uses query tokens to extract relevant information from visual features.

**Key characteristics:**
- Query-based feature extraction
- Better handling of complex visual scenes
- More sophisticated alignment between modalities

### Perceiver Projector

Derived from Google's Perceiver architecture, dynamically resamples visual features for better representation.

**Key characteristics:**
- Adaptive sampling of visual information
- Handles varying input sizes gracefully
- Good for complex visual understanding tasks

## Configuration Options

The enhanced builder provides multiple ways to configure your model:

### Dictionary-Based Configuration

```python
config = {
    "model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mm_vision_tower": "openai/clip-vit-large-patch14",
    "mm_hidden_size": 768,
    "encoder_type": "clip",  # Options: "clip", "sigclip", "biomedclip"
    "projector_type": "mlp",  # Options: "mlp", "identity", "qformer", "perceiver"
    "mm_projector_hidden_size": 1024
}

model, image_processor, tokenizer, context_len = load_pretrained_model(
    config,
    device="cuda"
)
```

### Object-Based Configuration

```python
from dataclasses import dataclass

@dataclass
class VisionConfig:
    mm_vision_tower: str = "openai/clip-vit-large-patch14"
    image_size: int = 224
    vision_feature_layer: int = -2

@dataclass
class ProjectorConfig:
    type: str = "mlp"
    hidden_size: int = 768
    num_hidden_layers: int = 2

model, image_processor, tokenizer, context_len = load_pretrained_model(
    model_path="path/to/model",
    custom_vision_config=VisionConfig(),
    projector_config=ProjectorConfig()
)
```

### Available Configuration Parameters

#### General Parameters
- `model_name_or_path`: Path to model or model name on HF Hub
- `model_base`: Base model for LoRA models
- `device`: Device to load model on (`cuda`, `cpu`, etc.)
- `load_8bit`: Whether to load in 8-bit mode
- `load_4bit`: Whether to load in 4-bit mode

#### Vision Encoder Parameters
- `mm_vision_tower`: Path to vision encoder model
- `encoder_type`: Type of encoder (`clip`, `sigclip`, `biomedclip`)
- `image_size`: Size of input images
- `vision_feature_layer`: Which layer to extract features from (-1 for last, -2 for second-to-last, etc.)
- `vision_feature_select_strategy`: How to select features (`default`, `patch`, `cls`)

#### Projector Parameters
- `projector_type`: Type of projector (`mlp`, `identity`, `qformer`, `perceiver`)
- `mm_projector_hidden_size`: Hidden size for projector layers
- `num_hidden_layers`: Number of layers in MLP projector
- `num_query_tokens`: Number of query tokens for QFormer
- `qformer_hidden_size`: Hidden size for QFormer
- `num_latents`: Number of latent vectors for Perceiver
- `latent_dim`: Dimension of latent vectors for Perceiver

## Usage Examples

### Basic Usage with Default Settings

```python
from moellava.model.builder_enhanced import load_pretrained_model

# Load a model with default settings (CLIP encoder, MLP projector)
model, image_processor, tokenizer, context_len = load_pretrained_model(
    model_path="path/to/model",
    device="cuda"
)
```

### Using SigClip Encoder

```python
# Using SigClip with automatic detection (based on model path)
model, image_processor, tokenizer, context_len = load_pretrained_model(
    model_path="path/to/model",
    device="cuda",
    custom_vision_config=VisionConfig(
        mm_vision_tower="google/siglip-base-patch16-224"
    )
)

# Explicitly specifying SigClip encoder
model, image_processor, tokenizer, context_len = load_pretrained_model(
    model_path="path/to/model",
    device="cuda",
    encoder_type="sigclip",
    custom_vision_config=VisionConfig(
        mm_vision_tower="google/siglip-base-patch16-224"
    )
)
```

### Using BiomedCLIP Encoder

```python
# Using BiomedCLIP with automatic detection
model, image_processor, tokenizer, context_len = load_pretrained_model(
    model_path="path/to/model",
    device="cuda",
    custom_vision_config=VisionConfig(
        mm_vision_tower="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
)

# Explicitly specifying BiomedCLIP encoder
model, image_processor, tokenizer, context_len = load_pretrained_model(
    model_path="path/to/model",
    device="cuda",
    encoder_type="biomedclip",
    custom_vision_config=VisionConfig(
        mm_vision_tower="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
)
```

### Specifying a QFormer Projector

```python
# Using a QFormer projector with dictionary configuration
config = {
    "model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mm_vision_tower": "openai/clip-vit-large-patch14",
    "mm_hidden_size": 768,
    "projector_type": "qformer",
    "num_query_tokens": 32,
    "qformer_hidden_size": 768
}

model, image_processor, tokenizer, context_len = load_pretrained_model(
    config,
    device="cuda"
)
```

### Using a Custom Projector Path

```python
# Load a model with a custom projector
model, image_processor, tokenizer, context_len = load_pretrained_model(
    model_path="path/to/model",
    projector_config={
        "path": "path/to/custom_projector.bin"
    }
)
```

## Implementation Details

### Encoders Package

The `encoders` package provides a framework for implementing and using custom vision encoders:

```
encoders/
├── __init__.py         # Base class and exports
├── factory.py          # Factory function for creating encoders
├── sigclip.py          # SigClip encoder implementation
└── biomedclip.py       # BiomedCLIP encoder implementation
```

The package uses a simple plugin architecture:

1. **Base Class**: `EncoderVisionTower` in `__init__.py` defines the interface all encoders must implement
2. **Factory Function**: `create_encoder()` in `factory.py` creates the appropriate encoder based on type
3. **Implementations**: Specific encoder implementations in dedicated files

Each encoder must implement:
- `load_model()`: Load the vision model and processor
- `to()`: Move model to specified device and dtype
- `forward()`: Process images through the vision model

### Adding a New Encoder

To add a new encoder:

1. Create a new file in the `encoders` folder (e.g., `my_encoder.py`)
2. Implement a class with the required methods
3. Update `__init__.py` to expose your encoder
4. Update `factory.py` to support your encoder type

Example for a hypothetical DINOv2 encoder:

```python
# encoders/dino.py
import torch
from transformers import AutoImageProcessor, AutoModel

class DINOEncoder:
    def __init__(self, model_name_or_path, config=None):
        self.model_name_or_path = model_name_or_path
        self.config = config
        self.is_loaded = False
        
    def load_model(self):
        self.model = AutoModel.from_pretrained(self.model_name_or_path)
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_name_or_path)
        self.is_loaded = True
        
    def to(self, **kwargs):
        if self.model is not None:
            self.model = self.model.to(**kwargs)
        return self
        
    def forward(self, images):
        if not self.is_loaded:
            self.load_model()
            
        outputs = self.model(images, output_hidden_states=True)
        # Extract the appropriate features based on config
        # ...
        return features
```

Then update `__init__.py` and `factory.py` accordingly.

## Testing

Several test scripts are included to verify functionality:

- `test_builder_enhanced.py`: Tests basic functionality
- `test_small_model.py`: Tests loading a small model end-to-end
- `test_sigclip_encoder.py`: Tests the SigClip encoder integration
- `test_biomedclip_encoder.py`: Tests the BiomedCLIP encoder integration

Run the tests with:

```bash
PYTHONPATH=/path/to/Med-MoE python test_builder_enhanced.py
```

## Troubleshooting

### Common Issues

#### Vision Encoder Issues

- **ImportError for encoder modules**: Ensure your PYTHONPATH includes the root directory
- **Missing dependencies**: Check that required packages are installed:
  ```bash
  # For SigClip
  pip install transformers>=4.34.0
  
  # For BiomedCLIP
  pip install open_clip_torch==2.23.0 transformers>=4.35.2
  ```
- **Model loading failures**: Verify model paths and internet connection for downloading

#### Projector Issues

- **Size mismatch errors**: Ensure projector dimensions match both encoder output and language model input
- **Missing projector files**: Check that custom projector paths exist
- **Incompatible projector types**: Some projectors may not be compatible with certain encoders

### Debugging Tips

- Set `projector_debug=True` in the configuration to get detailed logs about projector loading
- Set `encoder_debug=True` to get logs about encoder initialization and processing
- Check the model's `config.json` for existing projector/encoder configurations

## Extensions

### Creating Custom Projectors

Similar to custom encoders, you can implement custom projectors by:

1. Defining the projector architecture in a dedicated file
2. Updating the model loading code to support your projector type
3. Specifying the projector type in the configuration

Example for a custom gated projector:

```python
# In projectors/gated_projector.py
import torch.nn as nn

class GatedProjector(nn.Module):
    def __init__(self, vision_hidden_size, text_hidden_size):
        super().__init__()
        self.linear = nn.Linear(vision_hidden_size, text_hidden_size)
        self.gate = nn.Linear(vision_hidden_size, text_hidden_size)
        self.act = nn.Sigmoid()
        
    def forward(self, x):
        return self.linear(x) * self.act(self.gate(x))
```

### Future Improvements

- Add support for more vision encoder types (DINO, ViT, etc.)
- Implement a projector framework similar to the encoder framework
- Add batched processing support for more efficient inference
- Improve handling of quantized models with vision towers

## References

- [LLaVA Paper](https://arxiv.org/abs/2304.08485)
- [SigLip Paper](https://arxiv.org/abs/2303.15343)
- [BiomedCLIP Paper](https://arxiv.org/abs/2303.00915)
- [BLIP-2 Paper (QFormer reference)](https://arxiv.org/abs/2301.12597)
- [Perceiver Paper](https://arxiv.org/abs/2103.03206)

### Model Resources

- [SigLip Models on Hugging Face](https://huggingface.co/google/siglip-base-patch16-224)
- [BiomedCLIP Models on Hugging Face](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
- [OpenCLIP Documentation](https://github.com/mlfoundations/open_clip) 