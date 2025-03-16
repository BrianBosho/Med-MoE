# BiomedCLIP Integration Guide

This document explains how to use Microsoft's BiomedCLIP (Biomedical Contrastive Language-Image Pre-training) models as encoders in the enhanced MoELLaVA builder.

## Overview

BiomedCLIP is Microsoft's vision-language model specifically trained on biomedical data. It's designed to better understand medical images and text, making it particularly useful for medical vision-language tasks. This implementation allows you to use BiomedCLIP models as vision encoders in the MoELLaVA framework.

## Requirements

- `open_clip_torch` library (version 2.23.0 or higher)
- `transformers` library (version 4.35.2 or higher is recommended)
- Hugging Face Hub access for downloading pre-trained models
- MedVQA Enhanced Builder implementation

## Available BiomedCLIP Models

Microsoft has released several BiomedCLIP models on Hugging Face Hub:

- `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`: Base model with PubMedBERT text encoder
- `microsoft/BiomedCLIP-PubMedBERT_256-vit_large_patch14_224`: Large model with PubMedBERT text encoder
- `microsoft/BiomedCLIP-PubMedBERT_256-vit_large_patch14_336`: Large model with 336x336 input size

## Installation

Ensure you have the required packages installed:

```bash
pip install open_clip_torch==2.23.0 transformers==4.35.2
```

## Integration Components

The BiomedCLIP integration consists of the following components:

1. **BiomedClipEncoder Class**: A custom encoder implementation (`encoders/biomedclip.py`) that interfaces with `open_clip_torch` to handle loading, device placement, and image encoding using BiomedCLIP models.

2. **Factory Update**: The `create_encoder` function in `encoders/factory.py` was updated to support BiomedCLIP encoders.

3. **Builder Enhancement**: The `load_vision_encoder` function in `builder_enhanced.py` was updated to automatically detect BiomedCLIP models and use the appropriate processor.

## Usage

### Basic Usage

```python
from moellava.model.builder_enhanced import load_pretrained_model

# Using BiomedCLIP with a dictionary configuration
model_config = {
    "model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mm_vision_tower": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "mm_hidden_size": 768
}

# Load the model with BiomedCLIP encoder
model, image_processor, tokenizer, context_len = load_pretrained_model(
    model_config,
    encoder_type='biomedclip',
)
```

### Automatic Encoder Type Detection

The builder will automatically detect BiomedCLIP models based on the model path containing "biomedclip" or "medclip":

```python
model_config = {
    "model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mm_vision_tower": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "mm_hidden_size": 768
}

# No need to specify encoder_type - it will be detected automatically
model, image_processor, tokenizer, context_len = load_pretrained_model(
    model_config
)
```

### Custom Image Size

You can specify a custom image size using the `custom_vision_config`:

```python
from dataclasses import dataclass

# Define custom configuration
@dataclass
class CustomVisionConfig:
    image_size: int = 224
    vision_feature_layer: int = -2
    vision_feature_select_strategy: str = "default"

# Use the custom configuration
model, image_processor, tokenizer, context_len = load_pretrained_model(
    model_config,
    encoder_type='biomedclip',
    custom_vision_config=CustomVisionConfig()
)
```

## Testing

You can verify the BiomedCLIP integration using the provided test script:

```bash
PYTHONPATH=/path/to/Med-MoE python test_biomedclip_encoder.py
```

The test script checks both the direct loading of the BiomedCLIP encoder and the integration with a full model.

## Implementation Details

### Patch Features

Due to the way OpenCLIP exposes BiomedCLIP's features, the implementation generates artificial patch embeddings when patch mode is requested. This is a workaround since BiomedCLIP might not directly expose individual patch embeddings in the same way as the Transformers CLIP implementation.

### HF-Hub Format

BiomedCLIP models are loaded using OpenCLIP's `create_model_from_pretrained` function, which expects paths to be prefixed with `hf-hub:` when loading from Hugging Face. The implementation handles this automatically.

## Troubleshooting

- If you encounter the error `ModuleNotFoundError: No module named 'open_clip'`, make sure you have installed the `open_clip_torch` package.
- If the BiomedCLIP encoder fails to load, verify that your OpenCLIP library is updated to a version that supports BiomedCLIP models.
- If you receive warnings about model paths not containing "biomedclip", the implementation will still attempt to load the model but might fallback to a default BiomedCLIP model.

## Performance Considerations

- BiomedCLIP models are specifically trained on biomedical data and may perform better than general CLIP models on medical imagery.
- Consider using the larger models (e.g., `vit_large_patch14` variants) for more complex medical tasks.
- For best performance, use the image size that the model was trained on (typically 224x224 or 336x336).

## Limitations

- The current implementation focuses on the image encoder part of BiomedCLIP and doesn't utilize the specialized text encoder.
- Since we're using OpenCLIP's interface, some features available in Transformers' implementation might not be directly accessible.

## References

- [BiomedCLIP Paper: Large-scale Domain-specific Pretraining for Biomedical Vision-Language Processing](https://arxiv.org/abs/2303.00915)
- [BiomedCLIP Models on Hugging Face Hub](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
- [OpenCLIP Documentation](https://github.com/mlfoundations/open_clip) 