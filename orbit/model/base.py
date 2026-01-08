import torch
import torch.nn as nn
from typing import Union, List, Optional, Iterable

from orbit.utils import (
    auto_initialize,
    freeze_layers,
    unfreeze_layers,
    get_trainable_params,
    save_model,
    load_model,
)


class BaseBlock(nn.Module):
    def __init__(self):
        super(BaseBlock, self).__init__()

        self.gradient_checkpointing: bool = False
    
    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    def _init_weights(self, model: nn.Module):
        auto_initialize(model=model)
    
    def set_checkpoint(self, value: bool):
        self.gradient_checkpointing = value
        for model in self.modules():
            if isinstance(model, BaseBlock) and model is not self:
                model.gradient_checkpointing = value

    def count_params(self, trainable_only=False):
        if not trainable_only:
            return sum(p.numel() for p in self.parameters())
        

    def count_trainable_params(self) -> Iterable[torch.Tensor]:
        return get_trainable_params(self)

    def checkpoint(self, function, *args, **kwargs):
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(function, *args, use_reentrant=False, **kwargs)
        else:
            return function(*args, **kwargs)

    def freeze(self, targets: Optional[Union[str, List[str]]] = None):
        freeze_layers(self, targets)

    def unfreeze(self, targets: Optional[Union[str, List[str]]] = None):
        unfreeze_layers(self, targets)

    def save_pretrained(self, file_path: str):
        save_model(self, file_path)

    def load_pretrained(self, file_path: str, strict: bool = True, map_location: Union[str, torch.device] = 'cpu'):
        load_model(self, file_path, strict, map_location)
