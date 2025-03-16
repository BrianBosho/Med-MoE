d projecto# Enhanced Model Builder Guide

This comprehensive guide explains the enhanced model builder for MoELLaVA, including its modular architecture, pluggable vision encoders, and configurable projectors for multimodal models.

## Table of Contents

- [Overview](#overview)
- [Available Vision Encoders](#available-vision-encoders)
- [Available Multimodal Projectors](#available-multimodal-projectors)
- [Installation Requirements](#installation-requirements)
- [Usage Guide](#usage-guide)
  - [Basic Usage](#basic-usage)
  - [Using Dictionary Configurations](#using-dictionary-configurations)
  - [Custom Vision Configurations](#custom-vision-configurations)
  - [Selecting Vision Encoders](#selecting-vision-encoders)
  - [Specifying Projector Types](#specifying-projector-types)
- [Encoder Framework](#encoder-framework)
  - [Architecture](#architecture)
  - [Creating Custom Encoders](#creating-custom-encoders)
- [Implementation Details](#implementation-details)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Overview

The enhanced model builder for MoELLaVA provides a modular, flexible architecture for loading and configuring multimodal models. Key features include:

- **Modular Architecture**: Split into separate functions for loading configurations, tokenizers, vision encoders, and language models
- **Pluggable Vision Encoders**: Support for multiple vision encoders (CLIP, SigClip, BiomedCLIP) with automatic detection
- **Configurable Projectors**: Multiple projection architectures to transform vision features for language models
- **Enhanced Configuration System**: Support for both dictionary-based and object-based configuration
- **Comprehensive Model Support**: Compatible with LLaMA, Mistral, Phi, Qwen, MiniCPM, and StableLM variants

## Available Vision Encoders

| Encoder Type | Description | Recommended Use Case | Models |
|-------------|-------------|---------------------|--------|
| **CLIP** | OpenAI's Contrastive Language-Image Pre-training. The default vision encoder used in most LLaVA models. | General-purpose visual understanding | `openai/clip-vit-large-patch14`, `openai/clip-vit-base-patch16` |
| **SigClip** | Google's SigLip (Sigmoid Loss) vision-language model. Uses sigmoid loss for better zero-shot performance. | Zero-shot visual classification and recognition | `google/siglip-base-patch16-224`, `google/siglip-large-patch16-224` |
| **BiomedCLIP** | Microsoft's biomedical vision-language model trained on medical data. | Medical imaging and diagnostics | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` |

## Available Multimodal Projectors

| Projector Type | Description | Recommended Use Case |
|---------------|-------------|---------------------|
| **MLP** | Default projector using multiple linear layers with activations. | General-purpose projection, balanced performance |
| **Identity** | Pass-through projector without transformation. | When vision feature dimensions already match language model |
| **QFormer** | Query-based projection system similar to BLIP-2. | Complex reasoning tasks requiring better visual feature filtering |
| **Perceiver** | Cross-attention based module for flexible feature resampling. | Complex visual scenes with many objects |
| **LoRA** | Low-Rank Adaptation projector for fine-tuned models. | Fine-tuned models with custom projection weights |

## Installation Requirements

Basic installation requirements:
```bash
pip install torch transformers
```

For SigClip support:
```bash
# No additional requirements - uses standard transformers library
```

For BiomedCLIP support:
```bash
pip install open_clip_torch==2.23.0
```

## Usage Guide

### Basic Usage

```python
from moellava.model.builder_enhanced import load_pretrained_model

# Load a model with default settings (CLIP encoder and MLP projector)
model, image_processor, tokenizer, context_len = load_pretrained_model(
    model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device="cuda"
)
```

### Using Dictionary Configurations

```python
# Configure with a dictionary
config = {
    "model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mm_vision_tower": "openai/clip-vit-large-patch14",
    "mm_hidden_size": 768,
    "projector_type": "mlp",        # Options: "mlp", "identity", "qformer", "perceiver"
    "mm_projector_hidden_size": 1024
}

model, image_processor, tokenizer, context_len = load_pretrained_model(
    config,
    device="cuda"
)
```

### Custom Vision Configurations

```python
from dataclasses import dataclass

# Define a custom vision configuration
@dataclass
class VisionConfig:
    mm_vision_tower: str = "openai/clip-vit-large-patch14"
    image_size: int = 336
    vision_feature_layer: int = -2
    vision_feature_select_strategy: str = "default"

# Use the custom configuration
model, image_processor, tokenizer, context_len = load_pretrained_model(
    model_path="path/to/model",
    device="cuda",
    custom_vision_config=VisionConfig()
)
```

### Selecting Vision Encoders

#### CLIP (Default)

```python
# Using CLIP encoder (default)
model, image_processor, tokenizer, context_len = load_pretrained_model(
    model_path="path/to/model",
    device="cuda",
    custom_vision_config=VisionConfig(
        mm_vision_tower="openai/clip-vit-large-patch14"
    )
)

# OR explicitly specify encoder_type
model, image_processor, tokenizer, context_len = load_pretrained_model(
    model_path="path/to/model",
    device="cuda",
    encoder_type="clip",
    custom_vision_config=VisionConfig(
        mm_vision_tower="openai/clip-vit-large-patch14"
    )
)
```

#### SigClip

```python
# Using SigClip with automatic detection (based on model path)
model, image_processor, tokenizer, context_len = load_pretrained_model(
    model_path="path/to/model",
    device="cuda",
    custom_vision_config=VisionConfig(
        mm_vision_tower="google/siglip-base-patch16-224"
    )
)

# OR explicitly specify encoder_type
model, image_processor, tokenizer, context_len = load_pretrained_model(
    model_path="path/to/model",
    device="cuda",
    encoder_type="sigclip",
    custom_vision_config=VisionConfig(
        mm_vision_tower="google/siglip-base-patch16-224"
    )
)
```

#### BiomedCLIP

```python
# Using BiomedCLIP with automatic detection
model, image_processor, tokenizer, context_len = load_pretrained_model(
    model_path="path/to/model",
    device="cuda",
    custom_vision_config=VisionConfig(
        mm_vision_tower="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
)

# OR explicitly specify encoder_type
model, image_processor, tokenizer, context_len = load_pretrained_model(
    model_path="path/to/model",
    device="cuda",
    encoder_type="biomedclip",
    custom_vision_config=VisionConfig(
        mm_vision_tower="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
)
```

### Specifying Projector Types

#### MLP Projector (Default)

```python
# Using dictionary configuration
config = {
    "model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mm_vision_tower": "openai/clip-vit-large-patch14",
    "projector_type": "mlp",
    "mm_projector_hidden_size": 1024,
    "mm_projector_num_layers": 2
}

# OR using projector_config parameter
from dataclasses import dataclass

@dataclass
class ProjectorConfig:
    type: str = "mlp"
    hidden_size: int = 1024
    num_layers: int = 2
    dropout: float = 0.1

model, image_processor, tokenizer, context_len = load_pretrained_model(
    model_path="path/to/model",
    device="cuda",
    projector_config=ProjectorConfig()
)
```

#### QFormer Projector

```python
# Using dictionary configuration
config = {
    "model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mm_vision_tower": "openai/clip-vit-large-patch14",
    "projector_type": "qformer",
    "mm_qformer_num_query_tokens": 32,
    "mm_qformer_num_hidden_layers": 2
}

# Load the model with QFormer projector
model, image_processor, tokenizer, context_len = load_pretrained_model(
    config,
    device="cuda"
)
```

#### Perceiver Projector

```python
# Using dictionary configuration
config = {
    "model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mm_vision_tower": "openai/clip-vit-large-patch14",
    "projector_type": "perceiver",
    "mm_perceiver_num_latents": 64,
    "mm_perceiver_num_layers": 4
}

# Load the model with Perceiver projector
model, image_processor, tokenizer, context_len = load_pretrained_model(
    config,
    device="cuda"
)
```

#### Custom Projector Path

```python
# Load model with a custom pre-trained projector
model, image_processor, tokenizer, context_len = load_pretrained_model(
    model_path="path/to/model",
    device="cuda", 
    projector_config={
        "path": "path/to/custom_projector.bin"
    }
)
```

## Encoder Framework

### Architecture

The encoder framework in the `encoders` package provides a modular system for implementing different vision encoders. The framework consists of:

1. **Base Encoder Interface**: The `EncoderVisionTower` class in `encoders/__init__.py` defines the basic interface all encoders must implement:
   - `load_model()`: Loads the underlying vision model
   - `to()`: Moves the model to specified device/dtype
   - `forward()`: Processes images through the vision model

2. **Specific Encoder Implementations**:
   - `SigClipEncoder`: Implementation for Google's SigLip models
   - `BiomedClipEncoder`: Implementation for Microsoft's biomedical CLIP models

3. **Factory Module**: The `create_encoder` function in `encoders/factory.py` serves as a factory pattern implementation that instantiates the appropriate encoder based on the specified type or model path.

### Creating Custom Encoders

To implement your own custom encoder:

1. Create a new file in the `encoders` folder (e.g., `my_encoder.py`)
2. Implement a class that follows the encoder interface with `load_model()`, `to()`, and `forward()` methods
3. Update `encoders/__init__.py` to import and expose your encoder
4. Update `encoders/factory.py` to support your encoder type

Example custom encoder implementation:

```python
# In encoders/my_encoder.py
import torch

class MyCustomEncoder:
    def __init__(self, model_name_or_path, config=None):
        self.model_name_or_path = model_name_or_path
        self.config = config
        self.is_loaded = False
        self.model = None
        
    def load_model(self):
        """Load the vision model."""
        # Implementation for loading your custom model
        self.model = YourCustomModel.from_pretrained(self.model_name_or_path)
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
            
        # Implementation for processing images
        outputs = self.model(images)
        return outputs.features  # Return appropriate feature representation
```

## Implementation Details

### SigClip Integration

SigClip (Google's SigLip) models use sigmoid loss instead of the standard contrastive loss used in CLIP. The implementation handles these models by:

- Integrating with the Transformers library's native support for SigLip
- Using `AutoImageProcessor` for SigClip models instead of the standard CLIP processor
- Supporting custom image sizes through the processor configuration

### BiomedCLIP Integration

BiomedCLIP integration relies on the `open_clip_torch` library which requires special handling:

- Model paths are prefixed with `hf-hub:` when loading from Hugging Face
- Patch features are artificially created since OpenCLIP's interface doesn't expose intermediate hidden states
- The processor is obtained directly from the encoder after loading

### Projector Loading

Projectors are loaded in the `load_base_with_projector` function, which handles:

- Loading projector weights from the specified path or the default `mm_projector.bin`
- Supporting custom projector paths through the `projector_config` parameter
- Converting weights to FP16 for compatibility with the model

## Testing

Several test scripts are included to verify functionality:

- `test_builder_enhanced.py`: Tests basic builder functionality
- `test_small_model.py`: Tests loading a small model end-to-end
- `test_sigclip_encoder.py`: Tests the SigClip encoder integration
- `test_biomedclip_encoder.py`: Tests the BiomedCLIP encoder integration

Run the tests with:

```bash
PYTHONPATH=/path/to/Med-MoE python test_biomedclip_encoder.py
```

## Troubleshooting

### General Issues

- **AttributeError or KeyError**: Check that your configuration dictionary contains all required fields, or that your custom configuration object has the necessary attributes.
- **Out of Memory Errors**: Try reducing batch size or using a smaller model/encoder. Vision encoders can be memory-intensive.

### SigClip Issues

- If you receive warnings about missing processor configurations, these are normal and occur because SigLip image processors are relatively new in the Transformers library.
- Make sure your Transformers library is updated to a version that supports SigLip (4.34.0+).

### BiomedCLIP Issues

- `ModuleNotFoundError: No module named 'open_clip'`: Install the required package with `pip install open_clip_torch==2.23.0`
- If the BiomedCLIP encoder fails to load, verify that your OpenCLIP library is updated to a version that supports BiomedCLIP models.
- If you receive warnings about model paths not containing "biomedclip", the implementation will still attempt to load the model but might fallback to a default BiomedCLIP model.

### Projector Issues

- **Missing Projector Weights**: If the projector doesn't load correctly, check if the projector file exists at the specified path or in the model directory.
- **Dimension Mismatch**: If you encounter dimension mismatches, ensure that the projector's output dimension matches the language model's embedding dimension.

## References

- [LLaVA: Large Language and Vision Assistant](https://llava-vl.github.io/)
- [SigLip Paper: Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)
- [BiomedCLIP Paper: Large-scale Domain-specific Pretraining for Biomedical Vision-Language Processing](https://arxiv.org/abs/2303.00915)
- [BLIP-2 and QFormer Architecture](https://arxiv.org/abs/2301.12597)
- [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [OpenCLIP Documentation](https://github.com/mlfoundations/open_clip)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/) 