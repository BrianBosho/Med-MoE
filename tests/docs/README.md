# MoELLaVA Documentation

This directory contains documentation for the enhanced MoELLaVA builder and custom encoders.

## Main Documentation

The comprehensive documentation for the enhanced builder, including all encoder types and projectors, is available in:

**[ENHANCED_BUILDER_DOCS.md](./ENHANCED_BUILDER_DOCS.md)**

This file includes:
- Overview of the enhanced builder architecture
- Detailed descriptions of all supported vision encoders
- Comprehensive guide to multimodal projectors
- Configuration options and usage examples
- Implementation details and extension guidelines
- Troubleshooting tips and references

## Additional Documentation

| File | Description |
|------|-------------|
| `BIOMEDCLIP_INTEGRATION.md` | Specific details about BiomedCLIP integration (now consolidated in ENHANCED_BUILDER_DOCS.md) |
| `README_ENHANCED_BUILDER.md` | Original documentation for the enhanced builder (now consolidated in ENHANCED_BUILDER_DOCS.md) |

**Note:** The additional documentation files are kept for reference but their content has been consolidated into the main ENHANCED_BUILDER_DOCS.md file.

## Usage Example

Here's a quick example of how to use the enhanced builder with custom encoders:

```python
from moellava.model.builder_enhanced import load_pretrained_model

# Using dictionary configuration with BiomedCLIP encoder
config = {
    "model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mm_vision_tower": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "encoder_type": "biomedclip",
    "mm_hidden_size": 768,
    "projector_type": "mlp",
    "mm_projector_hidden_size": 1024
}

# Load the model with the configuration
model, image_processor, tokenizer, context_len = load_pretrained_model(
    config,
    device="cuda"
)
```

For more detailed examples and configuration options, please refer to the main documentation. 