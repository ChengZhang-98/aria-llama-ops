from pathlib import Path
import torch
from transformers import AutoConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

_MODEL_NAME = "Cheng98/TinyLlama_v1.1"


def load_tinyllama_cfg() -> LlamaConfig:
    config = AutoConfig.from_pretrained(_MODEL_NAME)
    return config


def load_tinyllama_ckpt() -> dict[str, torch.Tensor]:
    model_path = Path(snapshot_download(_MODEL_NAME, local_files_only=True))
    state_dict = load_file(model_path.joinpath("model.safetensors"))
    return state_dict
