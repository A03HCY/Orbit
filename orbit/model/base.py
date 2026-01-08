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
    ''' 基础模型块，提供通用的模型功能。

    继承自 nn.Module，包含参数统计、梯度检查点、冻结/解冻层、保存/加载模型等功能。
    '''

    def __init__(self):
        ''' 初始化 BaseBlock。 '''
        super(BaseBlock, self).__init__()

        self.gradient_checkpointing: bool = False
    
    @property
    def device(self):
        ''' 获取模型所在的设备。

        Returns:
            torch.device: 模型参数所在的设备。如果没有参数，则返回 cpu。
        '''
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    def _init_weights(self, model: nn.Module):
        ''' 初始化模型权重。

        Args:
            model (nn.Module): 需要初始化的模型。
        '''
        auto_initialize(model=model)
    
    def set_checkpoint(self, value: bool):
        ''' 设置是否启用梯度检查点。

        Args:
            value (bool): 是否启用梯度检查点。
        '''
        self.gradient_checkpointing = value
        for model in self.modules():
            if isinstance(model, BaseBlock) and model is not self:
                model.gradient_checkpointing = value

    def count_params(self, trainable_only=False):
        ''' 统计模型参数数量。

        Args:
            trainable_only (bool, optional): 是否只统计可训练参数。默认为 False。

        Returns:
            int: 参数数量。
        '''
        if trainable_only:
            return sum(p.numel() for p in get_trainable_params(self))
        
        return sum(p.numel() for p in self.parameters())

    def checkpoint(self, function, *args, **kwargs):
        ''' 应用梯度检查点。

        如果启用了梯度检查点且处于训练模式，则使用 torch.utils.checkpoint.checkpoint。
        否则直接调用函数。

        Args:
            function (Callable): 要执行的函数。
            *args: 传递给函数的位置参数。
            **kwargs: 传递给函数的关键字参数。

        Returns:
            Any: 函数的返回值。
        '''
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(function, *args, use_reentrant=False, **kwargs)
        else:
            return function(*args, **kwargs)

    def freeze(self, targets: Optional[Union[str, List[str]]] = None):
        ''' 冻结指定层的参数。

        Args:
            targets (Optional[Union[str, List[str]]], optional): 要冻结的层名称或名称列表。
                如果为 None，则冻结所有层。默认为 None。
        '''
        freeze_layers(self, targets)

    def unfreeze(self, targets: Optional[Union[str, List[str]]] = None):
        ''' 解冻指定层的参数。

        Args:
            targets (Optional[Union[str, List[str]]], optional): 要解冻的层名称或名称列表。
                如果为 None，则解冻所有层。默认为 None。
        '''
        unfreeze_layers(self, targets)

    def save_pretrained(self, file_path: str):
        ''' 保存模型权重到文件。

        Args:
            file_path (str): 保存路径。
        '''
        save_model(self, file_path)

    def load_pretrained(self, file_path: str, strict: bool = True, map_location: Union[str, torch.device] = 'cpu'):
        ''' 从文件加载模型权重。

        Args:
            file_path (str): 权重文件路径。
            strict (bool, optional): 是否严格匹配键值。默认为 True。
            map_location (Union[str, torch.device], optional): 映射位置。默认为 'cpu'。
        '''
        load_model(self, file_path, strict, map_location)
