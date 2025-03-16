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

import os
import warnings
import shutil
from peft import PeftModel,AutoPeftModel,AutoPeftModelForCausalLM
from moellava.model.language_model.llava_qwen_moe import EvalMoELLaVAQWenForCausalLM
from moellava.model.language_model.llava_qwen import LlavaQWenForCausalLM
from deepspeed.moe.layer import MoE

# from encoders import EncoderVisionTower

from moellava.model.language_model.llava_llama_moe import EvalMoELLaVALlamaForCausalLM
from moellava.model.language_model.llava_llama import LlavaLlamaForCausalLM

import transformers
a, b, c = transformers.__version__.split('.')[:3]
if a == '4' and int(b) >= 34:
    from moellava.model.language_model.llava_mistral_moe import EvalMoELLaVAMistralForCausalLM
    from moellava.model.language_model.llava_mistral import LlavaMistralForCausalLM
if a == '4' and int(b) >= 36:
    from moellava.model.language_model.llava_minicpm_moe import EvalMoELLaVAMiniCPMForCausalLM
    from moellava.model.language_model.llava_minicpm import LlavaMiniCPMForCausalLM
    from moellava.model.language_model.llava_phi_moe import EvalMoELLaVAPhiForCausalLM
    from moellava.model.language_model.llava_phi import LlavaPhiForCausalLM
    from moellava.model.language_model.llava_stablelm_moe import EvalMoELLaVAStablelmForCausalLM
    from moellava.model.language_model.llava_stablelm import LlavaStablelmForCausalLM
if a == '4' and int(b) >= 37:
    from moellava.model.language_model.llava_qwen1_5_moe import EvalMoELLaVAQwen1_5ForCausalLM
    from moellava.model.language_model.llava_qwen1_5 import LlavaQwen1_5ForCausalLM


from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, GenerationConfig
import torch
from moellava.model import *
from moellava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, \
    DEFAULT_VID_END_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VIDEO_PATCH_TOKEN
from moellava.model.language_model.qwen.tokenization_qwen import QWenTokenizer


def load_pretrained_model(model_path="", model_base=None, model_name=None, load_8bit=False, load_4bit=False, device_map="auto",
                          device="cuda", padding_side="right", merge=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    # Initialize processor at the beginning
    processor = {'image': None, 'video': None}
    
    # Ensure model_name is not None to avoid NoneType errors
    if model_name is None:
        import os
        model_name = os.path.basename(model_path.rstrip('/'))
        print(f"Warning: model_name was None, using '{model_name}' derived from model_path")
    
    # Configure model loading based on parameters
    if model_base:
        config_path = model_base
    else:
        config_path = model_path
        
    try:
        config = AutoConfig.from_pretrained(config_path)
        config.mm_vision_tower = "openai/clip-vit-large-patch14"
        if hasattr(config, 'mm_vision_tower') and config.mm_vision_tower is not None:
            # Check if custom_vision_config is provided to override vision tower
            custom_vision_config = kwargs.get('custom_vision_config', None)
            if custom_vision_config is not None and hasattr(custom_vision_config, 'mm_vision_tower'):
                vision_tower_path = custom_vision_config.mm_vision_tower
                print(f"Using custom vision tower: {vision_tower_path}")
                # Update config with the custom vision tower
                config.mm_vision_tower = vision_tower_path
                # Get encoder type from custom config
                encoder_type = getattr(custom_vision_config, 'encoder_type', None)
            else:
                # Use default encoder type from kwargs
                encoder_type = kwargs.get('encoder_type', None)
            
            # Try to load the encoder framework only when encoder_type is specified
            if encoder_type:
                try:
                    from encoders.factory import create_encoder
                    from encoders import EncoderVisionTower
                    print(f"Using custom {encoder_type} encoder for {config.mm_vision_tower}")
                    # Still load CLIP processor for compatibility
                    from transformers import CLIPImageProcessor
                    
                    # Check if custom_vision_config has an image_size attribute
                    custom_vision_config = kwargs.get('custom_vision_config', None)
                    processor_kwargs = {}
                    if custom_vision_config and hasattr(custom_vision_config, 'image_size'):
                        size = custom_vision_config.image_size
                        print(f"Using custom image size: {size} for image processor")
                        processor_kwargs['size'] = {"height": size, "width": size}
                    
                    # Create processor with custom size if specified
                    image_processor = CLIPImageProcessor.from_pretrained(
                        config.mm_vision_tower, 
                        **processor_kwargs
                    )
                except ImportError:
                    print("Encoder framework not found, falling back to default CLIP processor")
                    # use SigClip processor
                    from transformers import SigClipImageProcessor
                    image_processor = SigClipImageProcessor.from_pretrained(config.mm_vision_tower)
                    # from transformers import CLIPImageProcessor
                    # image_processor = CLIPImageProcessor.from_pretrained(config.mm_vision_tower)
                    print(f"Processor: {image_processor}")
            else:
                # Default behavior - use CLIP processor
                # default to sigclip processor
                from transformers import SigClipImageProcessor
                image_processor = SigClipImageProcessor.from_pretrained(config.mm_vision_tower)
                # from transformers import CLIPImageProcessor
                # image_processor = CLIPImageProcessor.from_pretrained(config.mm_vision_tower)
            processor['image'] = image_processor
    except Exception as e:
        print(f"Warning: Failed to load config or image processor: {e}")

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if model_name is not None and 'llava' in model_name.lower():
        # Load LLaVA model
        if model_name is not None and 'lora' in model_name.lower() and 'moe' not in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, padding_side=padding_side)
            print('Loading LLaVA from base model...')

            if 'qwen' in model_base.lower() and '1.5' not in model_base.lower():
                model = LlavaQWenForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
                model.generation_config = GenerationConfig.from_pretrained(model_base, pad_token_id=tokenizer.pad_token_id)
                # model.generation_config.repetition_penalty = None

                model.generation_config.do_sample = False  # use greedy decoding
                model.generation_config.repetition_penalty = 1.0  # disable repetition penalty
            elif 'openchat' in model_base.lower() or 'mistral' in model_base.lower():
                model = LlavaMistralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            elif 'qwen' in model_base.lower() and '1.5' in model_base.lower():
                model = LlavaQwen1_5ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
                model.config.eos_token_id = tokenizer.eos_token_id
            elif 'phi' in model_base.lower():
                model = LlavaPhiForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
                model.config.eos_token_id = tokenizer.eos_token_id
                # Add image processor initialization
                if hasattr(model.config, 'mm_vision_tower') and model.config.mm_vision_tower is not None:
                    from transformers import CLIPImageProcessor
                    image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower)
                    processor = {
                        'image': image_processor,
                        'video': None
                    }
                    
                    # Load and initialize the vision tower if it exists
                    if hasattr(model, 'get_image_tower'):
                        image_tower = model.get_image_tower()
                        if not image_tower.is_loaded:
                            image_tower.load_model()
                        image_tower.to(device=device, dtype=torch.float16)
            elif 'minicpm' in model_base.lower():
                model = LlavaMiniCPMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
                model.config.eos_token_id = tokenizer.eos_token_id
            elif 'stablelm' in model_base.lower():
                model = LlavaStablelmForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
                model.config.eos_token_id = tokenizer.eos_token_id
                print('changing stablelm to moe')       
                moe_layers_idx=list(range(24))[::2]
                for  layer_num in  moe_layers_idx:
                    model.model.layers[layer_num].mlp = HeirarchicalMoE(
                                dim = 2048,
                                mlp=model.model.layers[layer_num].mlp,
                                num_experts=(3,3),
                                capacity_factor_train =1.5,
                                capacity_factor_eval =2,
                            )             
                    
        
            else:
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            # =============================================================================================
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            # import ipdb
            # ipdb.set_trace()
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            import peft
            print('Loading LoRA weights...')
            model = peft.PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif 'lora' in model_name.lower() and 'moe' in model_name.lower():
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            print('Adapting to MoE...')
            if 'qwen' in model_name.lower() and '1.5' not in model_name.lower():
                tokenizer = QWenTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                model = EvalMoELLaVAQWenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
                model.generation_config = GenerationConfig.from_pretrained(model_path, pad_token_id=tokenizer.pad_token_id)
                # model.generation_config.repetition_penalty = None
                model.generation_config.do_sample = False  # use greedy decoding
                model.generation_config.repetition_penalty = 1.0  # disable repetition penalty
            elif 'openchat' in model_name.lower() or 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                model = EvalMoELLaVAMistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            elif 'qwen' in model_name.lower() and '1.5' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                model = EvalMoELLaVAQwen1_5ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                # import ipdb
                # ipdb.set_trace()
                model.config.eos_token_id = tokenizer.eos_token_id
            elif 'phi' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                model = EvalMoELLaVAPhiForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
                model.config.eos_token_id = tokenizer.eos_token_id
                # Add image processor initialization
                if hasattr(model.config, 'mm_vision_tower') and model.config.mm_vision_tower is not None:
                    from transformers import CLIPImageProcessor
                    image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower)
                    processor = {
                        'image': image_processor,
                        'video': None
                    }
                    
                    # Load and initialize the vision tower if it exists
                    if hasattr(model, 'get_image_tower'):
                        image_tower = model.get_image_tower()
                        if not image_tower.is_loaded:
                            image_tower.load_model()
                        image_tower.to(device=device, dtype=torch.float16)
            elif 'minicpm' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                model = EvalMoELLaVAMiniCPMForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
                model.config.eos_token_id = tokenizer.eos_token_id
            elif 'stablelm' in model_name.lower():
                from moellava.model.language_model.stablelm.tokenization_arcade100k import Arcade100kTokenizer
                #tokenizer = Arcade100kTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False,trust_remote_code=True, padding_side=padding_side)
                model = EvalMoELLaVAStablelmForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True,trust_remote_code=True,config=lora_cfg_pretrained, **kwargs)
                model.config.eos_token_id = tokenizer.eos_token_id
        
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False,trust_remote_code=True, padding_side=padding_side)
                model = EvalMoELLaVALlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            if not merge:
                import deepspeed
                deepspeed.init_distributed(dist_backend='nccl')
                # Initialize the DeepSpeed-Inference engine
                ds_engine = deepspeed.init_inference(model,
                                                     # mp_size=2,
                                                     # dtype=torch.half,
                                                     checkpoint=None,
                                                     replace_with_kernel_inject=True)
                model = ds_engine.module
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True, padding_side=padding_side)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            # =============================================================================================
            elif 'openchat' in model_name.lower() or 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, padding_side=padding_side)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                if getattr(cfg_pretrained, 'moe', {}).get('moe_enable', False):
                    model = EvalMoELLaVAMistralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                    import deepspeed
                    deepspeed.init_distributed(dist_backend='nccl')
                    # Initialize the DeepSpeed-Inference engine
                    ds_engine = deepspeed.init_inference(model,
                                                         # mp_size=2,
                                                         # dtype=torch.half,
                                                         checkpoint=None,
                                                         replace_with_kernel_inject=False)
                    model = ds_engine.module
                else:
                    model = LlavaMistralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            elif 'phi' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, padding_side=padding_side)
                cfg_pretrained = LlavaPhiConfig.from_pretrained(model_path)
                if getattr(cfg_pretrained, 'moe', {}).get('moe_enable', False):
                    model = EvalMoELLaVAPhiForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                    import deepspeed
                    #deepspeed.init_distributed(dist_backend='nccl')
                    deepspeed.init_distributed(dist_backend='nccl', port=29501)

                    # Initialize the DeepSpeed-Inference engine
                    ds_engine = deepspeed.init_inference(model,
                                                         # mp_size=2,
                                                         # dtype=torch.half,
                                                         checkpoint=None,
                                                         replace_with_kernel_inject=False)
                    model = ds_engine.module
                    # Add image processor initialization
                    if hasattr(model.config, 'mm_vision_tower') and model.config.mm_vision_tower is not None:
                        from transformers import CLIPImageProcessor
                        image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower)
                        processor = {
                            'image': image_processor,
                            'video': None
                        }
                        
                        # Load and initialize the vision tower if it exists
                        if hasattr(model, 'get_image_tower'):
                            image_tower = model.get_image_tower()
                            if not image_tower.is_loaded:
                                image_tower.load_model()
                            image_tower.to(device=device, dtype=torch.float16)
                else:
                    model = LlavaPhiForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                model.config.eos_token_id = tokenizer.eos_token_id
            elif 'qwen' in model_name.lower() and '1.5' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, padding_side=padding_side)
                cfg_pretrained = LlavaQwen1_5Config.from_pretrained(model_path)
                if getattr(cfg_pretrained, 'moe', {}).get('moe_enable', False):
                    model = EvalMoELLaVAQwen1_5ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                    import deepspeed
                    deepspeed.init_distributed(dist_backend='nccl')
                    # Initialize the DeepSpeed-Inference engine
                    ds_engine = deepspeed.init_inference(model,
                                                         # mp_size=2,
                                                         # dtype=torch.half,
                                                         checkpoint=None,
                                                         replace_with_kernel_inject=False)
                    model = ds_engine.module
                else:
                    model = LlavaQwen1_5ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                model.config.eos_token_id = tokenizer.eos_token_id
            elif 'minicpm' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, padding_side=padding_side)
                cfg_pretrained = LlavaMiniCPMConfig.from_pretrained(model_path)
                if getattr(cfg_pretrained, 'moe', {}).get('moe_enable', False):
                    model = EvalMoELLaVAMiniCPMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                    import deepspeed
                    deepspeed.init_distributed(dist_backend='nccl')
                    # Initialize the DeepSpeed-Inference engine
                    ds_engine = deepspeed.init_inference(model,
                                                         # mp_size=2,
                                                         # dtype=torch.half,
                                                         checkpoint=None,
                                                         replace_with_kernel_inject=False)
                    model = ds_engine.module
                else:
                    model = LlavaMiniCPMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                model.config.eos_token_id = tokenizer.eos_token_id
            elif 'stablelm' in model_name.lower():
                from moellava.model.language_model.stablelm.tokenization_arcade100k import Arcade100kTokenizer
                from moellava.model.language_model.stablelm.configuration_stablelm_epoch import StableLMEpochConfig
                tokenizer = Arcade100kTokenizer.from_pretrained(model_base, use_fast=False, padding_side=padding_side)
                cfg_pretrained = StableLMEpochConfig.from_pretrained(model_path)
                if getattr(cfg_pretrained, 'moe', {}).get('moe_enable', False):
                    model = EvalMoELLaVAStablelmForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                    import deepspeed
                    deepspeed.init_distributed(dist_backend='nccl')
                    # Initialize the DeepSpeed-Inference engine
                    ds_engine = deepspeed.init_inference(model,
                                                         # mp_size=2,
                                                         # dtype=torch.half,
                                                         checkpoint=None,
                                                         replace_with_kernel_inject=False)
                    model = ds_engine.module
                else:
                    model = LlavaStablelmForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                # model.config.eos_token_id = tokenizer.eos_token_id
            elif 'qwen' in model_name.lower() and '1.5' not in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, padding_side=padding_side)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                if getattr(cfg_pretrained, 'moe', {}).get('moe_enable', False):
                    model = EvalMoELLaVAQWenForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                    import deepspeed
                    deepspeed.init_distributed(dist_backend='nccl')
                    # Initialize the DeepSpeed-Inference engine
                    ds_engine = deepspeed.init_inference(model,
                                                         # mp_size=2,
                                                         # dtype=torch.half,
                                                         checkpoint=None,
                                                         replace_with_kernel_inject=False)
                    model = ds_engine.module
                else:
                    model = LlavaQWenForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                model.generation_config = GenerationConfig.from_pretrained(model_base, pad_token_id=tokenizer.pad_token_id)
                # model.generation_config.repetition_penalty = None

                model.generation_config.do_sample = False  # use greedy decoding
                model.generation_config.repetition_penalty = 1.0  # disable repetition penalty
        
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, padding_side=padding_side)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                if getattr(cfg_pretrained, 'moe', {}).get('moe_enable', False):
                    model = EvalMoELLaVALlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                    import deepspeed
                    deepspeed.init_distributed(dist_backend='nccl')
                    # Initialize the DeepSpeed-Inference engine
                    ds_engine = deepspeed.init_inference(model,
                                                         # mp_size=2,
                                                         # dtype=torch.half,
                                                         checkpoint=None,
                                                         replace_with_kernel_inject=False)
                    model = ds_engine.module
                else:
                    model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            # =============================================================================================

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                if 'moe' in model_name.lower():  # TODO: adapt to moe
                    raise NotImplementedError
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, padding_side=padding_side)
                    model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'qwen' in model_name.lower() and '1.5' not in model_name.lower():
                tokenizer = QWenTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                if 'moe' in model_name.lower():
                    assert not load_8bit and not load_4bit  # FIXME
                    model = EvalMoELLaVAQWenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                    import deepspeed
                    deepspeed.init_distributed(dist_backend='nccl')
                    # Initialize the DeepSpeed-Inference engine
                    ds_engine = deepspeed.init_inference(model,
                                                         # mp_size=2,
                                                         # dtype=torch.half,
                                                         checkpoint=None,
                                                         replace_with_kernel_inject=False)
                    model = ds_engine.module
                else:
                    model = LlavaQWenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                    print(model)
                model.generation_config = GenerationConfig.from_pretrained(model_path, pad_token_id=tokenizer.pad_token_id)
                # model.generation_config.repetition_penalty = None

                model.generation_config.do_sample = False  # use greedy decoding
                model.generation_config.repetition_penalty = 1.0  # disable repetition penalty
            elif 'openchat' in model_name.lower() or 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                # print(tokenizer)
                if 'moe' in model_name.lower():
                    assert not load_8bit and not load_4bit  # FIXME
                    model = EvalMoELLaVAMistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                    import deepspeed
                    deepspeed.init_distributed(dist_backend='nccl')
                    # Initialize the DeepSpeed-Inference engine
                    ds_engine = deepspeed.init_inference(model,
                                                         # mp_size=2,
                                                         # dtype=torch.half,
                                                         checkpoint=None,
                                                         replace_with_kernel_inject=False)
                    model = ds_engine.module
                else:
                    model = LlavaMistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                print(model)
            elif 'phi' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                # print(tokenizer)
                if 'moe' in model_name.lower():
                    assert not load_8bit and not load_4bit  # FIXME
                    model = EvalMoELLaVAPhiForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                    import deepspeed
                   
                    deepspeed.init_distributed(dist_backend='nccl')
                   
                    # Initialize the DeepSpeed-Inference engine
                    ds_engine = deepspeed.init_inference(model,
                                                         # mp_size=2,
                                                         # dtype=torch.half,
                                                         checkpoint=None,
                                                         replace_with_kernel_inject=False)
                    model = ds_engine.module
                    # Add image processor initialization
                    if hasattr(model.config, 'mm_vision_tower') and model.config.mm_vision_tower is not None:
                        from transformers import CLIPImageProcessor
                        image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower)
                        processor = {
                            'image': image_processor,
                            'video': None
                        }
                        
                        # Load and initialize the vision tower if it exists
                        if hasattr(model, 'get_image_tower'):
                            image_tower = model.get_image_tower()
                            if not image_tower.is_loaded:
                                image_tower.load_model()
                            image_tower.to(device=device, dtype=torch.float16)
                else:
                    model = LlavaPhiForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                    # Add image processor initialization
                    if hasattr(model.config, 'mm_vision_tower') and model.config.mm_vision_tower is not None:
                        from transformers import CLIPImageProcessor
                        image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower)
                        processor = {
                            'image': image_processor,
                            'video': None
                        }
                        
                        # Load and initialize the vision tower if it exists
                        if hasattr(model, 'get_image_tower'):
                            image_tower = model.get_image_tower()
                            if not image_tower.is_loaded:
                                image_tower.load_model()
                            image_tower.to(device=device, dtype=torch.float16)
                model.config.eos_token_id = tokenizer.eos_token_id
            
            elif 'qwen' in model_name.lower() and '1.5' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                # print(tokenizer)
                if 'moe' in model_name.lower():
                    assert not load_8bit and not load_4bit  # FIXME
                    model = EvalMoELLaVAQwen1_5ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                    import deepspeed
                    deepspeed.init_distributed(dist_backend='nccl')
                    # Initialize the DeepSpeed-Inference engine
                    ds_engine = deepspeed.init_inference(model,
                                                         # mp_size=2,
                                                         # dtype=torch.half,
                                                         checkpoint=None,
                                                         replace_with_kernel_inject=False)
                    model = ds_engine.module
                else:
                    model = LlavaQwen1_5ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                model.config.eos_token_id = tokenizer.eos_token_id
            elif 'minicpm' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                # print(tokenizer)
                if 'moe' in model_name.lower():
                    assert not load_8bit and not load_4bit  # FIXME
                    model = EvalMoELLaVAMiniCPMForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                    import deepspeed
                    deepspeed.init_distributed(dist_backend='nccl')
                    # Initialize the DeepSpeed-Inference engine
                    ds_engine = deepspeed.init_inference(model,
                                                         # mp_size=2,
                                                         # dtype=torch.half,
                                                         checkpoint=None,
                                                         replace_with_kernel_inject=False)
                    model = ds_engine.module
                else:
                    model = LlavaMiniCPMForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                model.config.eos_token_id = tokenizer.eos_token_id
            elif 'stablelm' in model_name.lower():
                print("aaaaa")
                from moellava.model.language_model.stablelm.tokenization_arcade100k import Arcade100kTokenizer
                #tokenizer = Arcade100kTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False,trust_remote_code=True, padding_side=padding_side)
                # print(tokenizer)
                if 'moe' in model_name.lower():
                    print("use moe")
                    assert not load_8bit and not load_4bit  # FIXME
                    model = EvalMoELLaVAStablelmForCausalLM.from_pretrained(model_path,low_cpu_mem_usage=True, **kwargs)
                    #model = AutoPeftModelForCausalLM.from_pretrained(model_path)

                    import deepspeed
                    deepspeed.init_distributed(dist_backend='nccl')
                    # Initialize the DeepSpeed-Inference engine
                    ds_engine = deepspeed.init_inference(model,
                                                         # mp_size=2,
                                                         # dtype=torch.half,
                                                         checkpoint=None,
                                                         replace_with_kernel_inject=False)
                    model = ds_engine.module
                else:
                    model = LlavaStablelmForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                # model.config.eos_token_id = tokenizer.eos_token_id
            
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                if 'moe' in model_name.lower():
                    assert not load_8bit and not load_4bit  # FIXME
                    model = EvalMoELLaVALlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                    import deepspeed
                    deepspeed.init_distributed(dist_backend='nccl')
                    print(model)
                    # Initialize the DeepSpeed-Inference engine
                    ds_engine = deepspeed.init_inference(model,
                                                         # mp_size=2,
                                                         # dtype=torch.half,
                                                         checkpoint=None,
                                                         replace_with_kernel_inject=False)
                    model = ds_engine.module
                else:
                    model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, padding_side=padding_side)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, padding_side=padding_side)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    # ==========================================================================================================
    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        if model.config.mm_image_tower is not None:
            image_tower = model.get_image_tower()
            if not image_tower.is_loaded:
                image_tower.load_model()
            image_tower.to(device=device, dtype=torch.float16)
            image_processor = image_tower.image_processor
            processor['image'] = image_processor

        if model.config.mm_video_tower is not None:
            video_tower = model.get_video_tower()
            if not video_tower.is_loaded:
                video_tower.load_model()
            video_tower.to(device=device, dtype=torch.float16)
            video_processor = video_tower.video_processor
            processor['video'] = video_processor

    # ==========================================================================================================
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    # Add fallback for image processor if it's still None
    if processor['image'] is None:
        try:
            print("Image processor is None, trying to load default CLIP tower as fallback On the final Section")
            
            # Create a vision tower configuration
            vision_tower_name = "openai/clip-vit-large-patch14"
            
            # Check if custom_vision_config is provided
            custom_vision_config = kwargs.get('custom_vision_config', None)
            if custom_vision_config:
                # Use the custom vision tower if provided
                if hasattr(custom_vision_config, 'mm_vision_tower'):
                    vision_tower_name = custom_vision_config.mm_vision_tower
                    print(f"Using custom vision tower: {vision_tower_name} for fallback")
            
            # Create a configuration object with necessary attributes
            class VisionConfig:
                def __init__(self):
                    self.mm_vision_select_layer = -2  # Default value
                    self.mm_vision_select_feature = 'patch'  # Default value
                    self.encoder_type = kwargs.get('encoder_type', None)
                    
                    # Copy image_size from custom_vision_config if available
                    if custom_vision_config and hasattr(custom_vision_config, 'image_size'):
                        self.image_size = custom_vision_config.image_size
                        print(f"Using custom image size: {self.image_size} for vision config")
            
            vision_config = VisionConfig()
            
            # Use encoder framework if specified and available
            if kwargs.get('encoder_type', None):
                try:
                    from encoders import EncoderVisionTower
                    print(f"Using EncoderVisionTower with {kwargs['encoder_type']} encoder")
                    vision_tower = EncoderVisionTower(vision_tower_name, vision_config)
                    vision_tower.load_model()
                    vision_tower.to(device=device, dtype=torch.float16)
                    
                    # We still need a CLIP processor for compatibility
                    from transformers import CLIPImageProcessor
                    processor['image'] = CLIPImageProcessor.from_pretrained(vision_tower_name)
                except ImportError:
                    # Fall back to default CLIP if encoder framework not available
                    print("Encoder framework not available, falling back to CLIP")
                    from transformers import CLIPImageProcessor, CLIPVisionModel
                    from moellava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
                    vision_tower = CLIPVisionTower(vision_tower_name, args=vision_config)
                    vision_tower.load_model()
                    vision_tower.to(device=device, dtype=torch.float16)
                    processor['image'] = vision_tower.image_processor
            else:
                # Use default CLIP vision tower
                from transformers import CLIPImageProcessor, CLIPVisionModel
                from moellava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
                vision_tower = CLIPVisionTower(vision_tower_name, args=vision_config)
                vision_tower.load_model()
                vision_tower.to(device=device, dtype=torch.float16)
                processor['image'] = vision_tower.image_processor
            
            # Attach the vision tower to the model if it has the right methods
            if hasattr(model, 'get_model') and hasattr(model, 'get_image_tower'):
                # Set the vision tower
                if 'llava' in model_name.lower() and 'phi' in model_name.lower():
                    print("Attaching vision tower to model")
                    setattr(model.get_model(), 'image_tower', vision_tower)
                    # Also set the config
                    if not hasattr(model.config, 'mm_vision_tower') or model.config.mm_vision_tower is None:
                        model.config.mm_vision_tower = vision_tower_name
            
            print("Successfully loaded fallback vision tower and processor")
        except Exception as e:
            print(f"Warning: Failed to load fallback vision tower and processor: {e}")

    return tokenizer, model, processor, context_len
