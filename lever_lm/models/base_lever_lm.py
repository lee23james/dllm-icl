"""Base class for Lever LM (e.g. GPT2LeverLM)."""
from typing import List, Optional

import torch
from torch import nn


class BaseLeverLM(nn.Module):
    """Base for ICD sequence generation models: adapter/norm flags and freeze_prefix."""

    def __init__(
        self,
        adapter: bool = False,
        norm: bool = False,
        query_encoding_flag: Optional[List[str]] = None,
        icd_encoding_flag: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self._adapter = adapter
        self._norm = norm
        self.query_encoding_flag = query_encoding_flag or []
        self.icd_encoding_flag = icd_encoding_flag or []

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclass must implement forward()")

    def freeze_prefix(self, freeze_prefix_list: Optional[List[str]] = None) -> None:
        """Freeze parameters whose name starts with any of the given prefixes."""
        if not freeze_prefix_list:
            return
        for name, param in self.named_parameters():
            for prefix in freeze_prefix_list:
                if name.startswith(prefix):
                    param.requires_grad = False
                    break

    @torch.inference_mode()
    def generation(self, *args, **kwargs):
        raise NotImplementedError("Subclass must implement generation()")

