import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, image_tower, args, delay_load=False, cache_dir='./cache_dir'):
        super().__init__()

        self.is_loaded = False

        self.image_tower_name = image_tower
        self.image_tower = image_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        self.cache_dir = cache_dir

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.image_tower_name, cache_dir=self.cache_dir)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.image_tower_name, cache_dir=self.cache_dir)
        self.image_tower = CLIPVisionModel.from_pretrained(self.image_tower_name, cache_dir=self.cache_dir)
        self.image_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if not self.is_loaded:
            self.load_model()
        
        # Get expected image size from the config
        expected_image_size = self.config.image_size
        
        if type(images) is list:
            image_features = []
            for image in images:
                # Check and resize image if necessary
                if (image.shape[-2] != expected_image_size or 
                    image.shape[-1] != expected_image_size):
                    # Log warning about resizing
                    print(f"Warning: Resizing image from {image.shape[-2]}x{image.shape[-1]} to {expected_image_size}x{expected_image_size}")
                    # Use interpolate to resize the image
                    image = torch.nn.functional.interpolate(
                        image.unsqueeze(0),  # Add batch dimension
                        size=(expected_image_size, expected_image_size),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)  # Remove batch dimension
                    
                # Ensure image is on the correct device and dtype
                device_image = image.to(device=self.device, dtype=self.dtype)
                image_forward_out = self.image_tower(device_image.unsqueeze(0), output_hidden_states=True)
                # Maintain the device of the output
                image_feature = self.feature_select(image_forward_out).to(device=self.device, dtype=image.dtype)
                image_features.append(image_feature)
            return image_features
        else:
            # Check and resize images if necessary
            if (images.shape[-2] != expected_image_size or 
                images.shape[-1] != expected_image_size):
                # Log warning about resizing
                print(f"Warning: Resizing images from {images.shape[-2]}x{images.shape[-1]} to {expected_image_size}x{expected_image_size}")
                # Use interpolate to resize the image batch
                images = torch.nn.functional.interpolate(
                    images,
                    size=(expected_image_size, expected_image_size),
                    mode='bilinear',
                    align_corners=False
                )
                
            # Ensure images are on the correct device and dtype
            device_images = images.to(device=self.device, dtype=self.dtype)
            image_forward_outs = self.image_tower(device_images, output_hidden_states=True)
            # Maintain the device of the output
            image_features = self.feature_select(image_forward_outs).to(device=self.device, dtype=images.dtype)
            return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.image_tower.dtype

    @property
    def device(self):
        return self.image_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.image_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
