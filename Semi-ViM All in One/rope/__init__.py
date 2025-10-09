"""
Lightweight rotary embedding helpers used by the Vision Mamba backbone.
"""

import torch


class VisionRotaryEmbeddingFast:
    """Fallback rotary embedding that returns the original tensor."""

    def __init__(self, dim, pt_seq_len=None, ft_seq_len=None):
        self.dim = dim
        self.pt_seq_len = pt_seq_len
        self.ft_seq_len = ft_seq_len

    def __call__(self, x):
        return x


__all__ = ["VisionRotaryEmbeddingFast"]
