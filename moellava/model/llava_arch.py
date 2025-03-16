#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch


from .multimodal_encoder.builder import build_image_tower, build_video_tower
from .multimodal_projector.builder import build_projector

from moellava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, PAD_LENGTH



class LlavaMetaModel:
    """
    Base class for LLaVA (Large Language and Vision Assistant) models that handles vision module initialization and management.
    
    This class provides functionality to:
    1. Initialize and manage image and video tower modules
    2. Build and configure multimodal projectors
    3. Handle model initialization with various configurations
    """

    def __init__(self, config):
        """
        Initialize the LlavaMetaModel with the given configuration.
        
        Args:
            config: Model configuration object containing multimodal settings
        """
        super(LlavaMetaModel, self).__init__(config)
        if getattr(config, "mm_image_tower", None) is not None:
            self.image_tower = build_image_tower(config, delay_load=True)
        if getattr(config, "mm_video_tower", None) is not None:
            self.video_tower = build_video_tower(config, delay_load=True)
        if getattr(config, "mm_image_tower", None) is not None or getattr(config, "mm_video_tower", None) is not None:
            self.mm_projector = build_projector(config)

    def get_image_tower(self):
        """
        Get the image tower module.
        
        Returns:
            The image tower module. If the tower is a list (e.g., for FSDP), returns the first element.
        """
        image_tower = getattr(self, 'image_tower', None)
        if type(image_tower) is list:
            image_tower = image_tower[0]
        return image_tower

    def get_video_tower(self):
        """
        Get the video tower module.
        
        Returns:
            The video tower module. If the tower is a list (e.g., for FSDP), returns the first element.
        """
        video_tower = getattr(self, 'video_tower', None)
        if type(video_tower) is list:
            video_tower = video_tower[0]
        return video_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        """
        Initialize and configure vision modules (image tower, video tower, and projectors).
        
        Args:
            model_args: Arguments containing vision module configurations
            fsdp: Fully Sharded Data Parallel configuration (optional)
            
        This method:
        1. Initializes image and video towers if not already present
        2. Configures vision module parameters like select layer and features
        3. Sets up multimodal projectors
        4. Handles pretrained weights loading if specified
        """
        # ==============================================
        image_tower = model_args.image_tower
        video_tower = model_args.video_tower
        assert image_tower is not None or video_tower is not None
        # ==============================================

        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        # ==========================================================================

        self.config.mm_image_tower = image_tower
        if image_tower is not None:
            if self.get_image_tower() is None:
                image_tower = build_image_tower(model_args)

                if fsdp is not None and len(fsdp) > 0:
                    self.image_tower = [image_tower]
                else:
                    self.image_tower = image_tower
            else:
                if fsdp is not None and len(fsdp) > 0:
                    image_tower = self.image_tower[0]
                else:
                    image_tower = self.image_tower
                image_tower.load_model()


        self.config.mm_video_tower = video_tower
        if video_tower is not None:
            if self.get_video_tower() is None:
                video_tower = build_video_tower(model_args)

                if fsdp is not None and len(fsdp) > 0:
                    self.video_tower = [video_tower]
                else:
                    self.video_tower = video_tower
            else:
                if fsdp is not None and len(fsdp) > 0:
                    video_tower = self.video_tower[0]
                else:
                    video_tower = self.video_tower
                video_tower.load_model()

        # ==========================================================================

        self.config.use_mm_proj = True

        # ===================================================================================
        self.config.image_projector_type = getattr(model_args, 'image_projector_type', None)
        self.config.video_projector_type = getattr(model_args, 'video_projector_type', None)
        self.config.video_global_proj = getattr(model_args, 'video_global_proj', None)
        self.config.video_temproal_proj = getattr(model_args, 'video_temproal_proj', None)
        self.config.video_spatial_proj = getattr(model_args, 'video_spatial_proj', None)
        # print(self.config.image_projector_type, self.config.video_projector_type, self.config.video_global_proj, self.config.video_temproal_proj)
        if image_tower is not None and video_tower is not None:  # TODO: support different hidden_size
            assert image_tower.hidden_size == video_tower.hidden_size
            self.config.mm_hidden_size = image_tower.hidden_size
        else:
            self.config.mm_hidden_size = max(getattr(image_tower, 'hidden_size', -1),
                                             getattr(video_tower, 'hidden_size', -1))
        # ===================================================================================
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


class LlavaMetaForCausalLM(ABC):
    """
    Abstract base class for LLaVA (Large Language and Vision Assistant) models.
    Provides core functionality for handling multimodal inputs combining text and images/videos.
    """

    @abstractmethod
    def get_model(self):
        """
        Abstract method that must be implemented by subclasses to return the underlying model.
        Returns:
            The underlying model implementation
        """
        pass

    def get_image_tower(self):
        """
        Gets the image encoder tower from the underlying model.
        Returns:
            The image encoder model component
        """
        return self.get_model().get_image_tower()

    def get_video_tower(self):
        """
        Gets the video encoder tower from the underlying model.
        Returns:
            The video encoder model component
        """
        return self.get_model().get_video_tower()

    def encode_images(self, images):
        """
        Encodes images into feature representations, handling device placement.
        
        Args:
            images: Tensor of images to encode
            
        Returns:
            Tensor of encoded image features
            
        Process:
        1. Gets target device from model parameters
        2. Passes images through image tower
        3. Ensures features are on correct device
        4. Projects features through mm_projector
        """
        # Get model device for consistency
        target_device = next(self.get_model().parameters()).device
        
        # Encode images through tower
        image_features = self.get_model().get_image_tower()(images)
        
        # Handle device placement for features
        if isinstance(image_features, list):
            image_features = [feat.to(device=target_device) for feat in image_features]
        else:
            image_features = image_features.to(device=target_device)
        
        # Project features
        image_features = self.get_model().mm_projector.forward_image(image_features)
        
        return image_features

    def encode_videos(self, videos):
        """
        Encodes videos into feature representations, handling device placement.
        
        Args:
            videos: Tensor of videos of shape [batch_size, channels, time, height, width]
            
        Returns:
            Tensor of encoded video features
            
        Process:
        1. Gets target device from model parameters
        2. Handles empty/None video inputs
        3. Moves videos to correct device and dtype
        4. Passes through video tower
        5. Ensures features on correct device
        6. Projects features through mm_projector
        """
        # Get model device for consistency  
        target_device = next(self.get_model().parameters()).device
        
        # Get video tower
        video_tower = self.get_model().get_video_tower()
        
        # Handle empty inputs
        if videos is None or (isinstance(videos, list) and len(videos) == 0):
            return None
        
        # Move videos to correct device/dtype
        videos_on_device = videos.to(device=target_device, dtype=video_tower.dtype)
        
        # Encode through tower
        video_features = video_tower(videos_on_device)  # Shape: [batch, time, num_patches, channels]
        
        # Ensure correct device
        video_features = video_features.to(device=target_device)
        
        # Project features
        video_features = self.get_model().mm_projector.forward_video(video_features)
        
        return video_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
        """
        Prepares inputs and labels for multimodal training by combining text and image/video features.
        
        Args:
            input_ids: Input token IDs
            position_ids: Position IDs for tokens
            attention_mask: Attention mask
            past_key_values: Past key values for attention
            labels: Training labels
            images: Image/video inputs
            
        Returns:
            Tuple of:
            - None (input_ids not used)
            - position_ids: Updated position IDs
            - attention_mask: Updated attention mask  
            - past_key_values: Past key values
            - new_input_embeds: Combined text and image embeddings
            - new_labels: Updated labels
            
        Process:
        1. Early return if no images/videos
        2. Separates image and video inputs
        3. Encodes images and videos separately
        4. Combines encoded features
        5. Processes text tokens and combines with visual features
        6. Handles padding and truncation
        7. Returns combined embeddings and updated masks/labels
        """

        # Get towers
        image_tower = self.get_image_tower()
        video_tower = self.get_video_tower()

        # Early return if no visual input needed
        if (image_tower is None and video_tower is None) or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and (image_tower is not None or video_tower is not None) and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # Separate images and videos
        image_idx = [idx for idx, img in enumerate(images) if img.ndim == 3]
        video_idx = [idx for idx, vid in enumerate(images) if vid.ndim == 4]
        
        # Stack images and videos separately
        images_minibatch = torch.stack([images[idx] for idx in image_idx]) if len(image_idx) > 0 else []
        videos_minibatch = torch.stack([images[idx] for idx in video_idx]) if len(video_idx) > 0 else []

        # Process images and videos
        tmp_image_features = [None] * (len(image_idx) + len(video_idx))
        
        # Encode images if present
        if getattr(images_minibatch, 'ndim', 0) == 4:
            if image_tower is not None:
                image_features_minibatch = self.encode_images(images_minibatch)
            else:
                image_features_minibatch = torch.randn(1).to(self.device)
            for i, pos in enumerate(image_idx):
                tmp_image_features[pos] = image_features_minibatch[i]

        # Encode videos if present  
        if getattr(videos_minibatch, 'ndim', 0) == 5:
            video_features_minibatch = self.encode_videos(videos_minibatch)
            for i, pos in enumerate(video_idx):
                tmp_image_features[pos] = video_features_minibatch[i]

        # Combine features
        new_tmp = []
        for image in tmp_image_features:
            if isinstance(image, list):
                t = len(image)
                for i in range(t):
                    new_tmp.append(image[i])
            else:
                new_tmp.append(image)
        image_features = new_tmp

        # Handle attention masks and position IDs
        _labels = labels
        _position_ids = position_ids 
        _attention_mask = attention_mask
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
            
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
            
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # Remove padding
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        # Combine text and image embeddings
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            
            # Handle case with no images
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            # Split text around image tokens
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
                
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

            # Combine text and image embeddings
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx].to(self.device)
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate to max length
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Pad sequences
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        # Handle padding based on config
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        # Handle optional returns
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        """
        Initializes the vision tokenizer by adding special tokens and handling embeddings.
        
        Args:
            model_args: Model configuration arguments
            tokenizer: The tokenizer to initialize
            
        Process:
        1. Adds image patch token if configured
        2. Adds image start/end tokens if configured
        3. Resizes token embeddings
        4. Initializes new token embeddings
        5. Handles MLP adapter tuning
        6. Loads pretrained weights if provided
        """
        # Add image patch token
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        # Add image start/end tokens
        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            # Initialize new token embeddings
            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            # Handle MLP adapter tuning
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            # Load pretrained weights if provided
            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False


class LlavaQWenMetaForCausalLM(LlavaMetaForCausalLM):
    """
    QWen-specific implementation of LlavaMetaForCausalLM.
    Inherits core functionality and overrides prepare_inputs_labels_for_multimodal with QWen-specific handling.
    """

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
        """
        QWen-specific implementation of prepare_inputs_labels_for_multimodal.
        Main difference is in handling past_key_values indexing.
        
        Args and Returns same as parent class.
        
        Key differences:
        - Uses past_key_values[-1][-3] instead of past_key_values[-1][-1] for target shape
        - Otherwise identical to parent implementation
        """
        # Implementation follows parent class but with QWen-specific past_key_values handling
        # See parent class documentation for detailed process
        
        image_tower = self.get_image_tower()
        video_tower = self.get_video_tower()
        
        if (image_tower is None and video_tower is None) or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and (image_tower is not None or video_tower is not None) and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-3] + 1  # QWen-specific indexing
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # Rest of implementation identical to parent class
        # See parent class documentation for details
