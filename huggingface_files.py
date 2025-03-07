from huggingface_hub import hf_hub_download
import os
import glob

repo_id = "JsST/Med-MoE"
directory = "stage3/llavaphi-2.7b-medmoe"
local_dir = "./MedMoE-phi2"



# List of expected filenames (adjust based on actual names)
filenames = [
    # "added_tokens.json",
    # "config.json",
    # "generation_config.json",
    # "model.safetensors.index.json",
    # "special_tokens_map.json",
    # "tokenizer_config.json",
    # "trainer_state.json",
    # "vocab.json"
    "merges.txt",
]

os.makedirs(local_dir, exist_ok=True)
for filename in filenames:
    hf_hub_download(
        repo_id=repo_id,
        filename=f"{directory}/{filename}",
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )