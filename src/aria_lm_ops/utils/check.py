import numpy as np
import torch


def check_shape(x: torch.Tensor, shape: tuple[int]):
    assert x.ndim == len(shape), f"Expected {len(shape)} dimensions, got {x.ndim}"
    assert x.size() == shape, f"Expected shape {shape}, got {x.size()}"
