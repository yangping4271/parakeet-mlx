import json

import mlx.core as mx
from dacite import from_dict
from huggingface_hub import hf_hub_download
from mlx.utils import tree_flatten, tree_unflatten

from parakeet_mlx.parakeet import ParakeetTDT, ParakeetTDTArgs


def from_config(config: dict) -> ParakeetTDT:
    """Loads model from config (randomized weight)"""
    cfg = from_dict(ParakeetTDTArgs, config)  # TODO: model classification logic
    model = ParakeetTDT(cfg)

    return model


def from_pretrained(hf_id: str, dtype: mx.Dtype = mx.bfloat16) -> ParakeetTDT:
    """Loads model from Hugging Face"""
    config = json.load(open(hf_hub_download(hf_id, "config.json"), "r"))
    weight = hf_hub_download(hf_id, "model.safetensors")

    model = from_config(config)
    model.load_weights(weight)

    # cast dtype
    curr_weights = dict(tree_flatten(model.parameters()))
    curr_weights = [(k, v.astype(dtype)) for k, v in curr_weights.items()]
    model.update(tree_unflatten(curr_weights))

    return model
