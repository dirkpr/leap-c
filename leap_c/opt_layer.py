"""This provides the abstract interface for optimization layers."""
from abc import ABC, abstractmethod
from typing import TypeVar,  Generic

import torch.nn as nn


OptInput = TypeVar('OptInput')
OptState = TypeVar('OptState')
OptOutput= TypeVar('OptOutput')


class OptLayer(nn.Module, ABC, Generic[OptInput, OptState, OptOutput]):
    """A PyTorch module for an implicit optimization layer."""

    @abstractmethod
    def forward(self, opt_input: OptInput, opt_state: OptState) -> tuple[OptOutput, OptState, dict[str, float]]:
        ...
