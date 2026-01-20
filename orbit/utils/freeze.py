import torch
import torch.nn as nn
from typing import Union, List, Optional, Iterable, Type
from dataclasses import dataclass

@dataclass
class ParamStats:
    count: int
    params: List[torch.Tensor]

    def __iter__(self):
        return iter(self.params)

def set_trainable(
    model: nn.Module, 
    targets: Optional[Union[str, List[str]]] = None, 
    target_classes: Optional[Union[Type[nn.Module], List[Type[nn.Module]]]] = None,
    trainable: bool = False
) -> None:
    '''设置模型参数的 requires_grad 属性，用于冻结或解冻层。

    Args:
        model (nn.Module): 目标模型。
        targets (str or List[str], optional): 要操作的层名称或参数名称模式。
            - 如果为 None 且 target_classes 也为 None，则操作模型的所有参数。
            - 如果为 str，则操作名称中包含该字符串的所有参数。
            - 如果为 List[str]，则操作名称中包含列表中任意字符串的所有参数。
        target_classes (Type[nn.Module] or List[Type[nn.Module]], optional): 要操作的模块类。
            - 如果指定，则操作属于该类（或其子类）的所有模块的参数。
        trainable (bool): 是否可训练 (True 为解冻, False 为冻结)。
    '''
    if targets is None and target_classes is None:
        for param in model.parameters():
            param.requires_grad = trainable
        return

    if targets is not None:
        if isinstance(targets, str):
            targets = [targets]
        
        for name, param in model.named_parameters():
            if any(t in name for t in targets): param.requires_grad = trainable

    if target_classes is not None:
        if not isinstance(target_classes, (list, tuple)): target_classes = [target_classes]
        
        target_classes = tuple(target_classes)

        for module in model.modules():
            if isinstance(module, target_classes):
                for param in module.parameters(): param.requires_grad = trainable

def freeze_layers(
    model: nn.Module, 
    targets: Optional[Union[str, List[str]]] = None,
    target_classes: Optional[Union[Type[nn.Module], List[Type[nn.Module]]]] = None
) -> None:
    '''冻结模型指定层或所有层 (requires_grad=False)。

    Args:
        model (nn.Module): 目标模型。
        targets (str or List[str], optional): 要冻结的层名称模式。
        target_classes (Type[nn.Module] or List[Type[nn.Module]], optional): 要冻结的模块类。
        如果不指定 targets 和 target_classes，则冻结整个模型。
    '''
    set_trainable(model, targets, target_classes, trainable=False)

def unfreeze_layers(
    model: nn.Module, 
    targets: Optional[Union[str, List[str]]] = None,
    target_classes: Optional[Union[Type[nn.Module], List[Type[nn.Module]]]] = None
) -> None:
    '''解冻模型指定层或所有层 (requires_grad=True)。

    Args:
        model (nn.Module): 目标模型。
        targets (str or List[str], optional): 要解冻的层名称模式。
        target_classes (Type[nn.Module] or List[Type[nn.Module]], optional): 要解冻的模块类。
        如果不指定 targets 和 target_classes，则解冻整个模型。
    '''
    set_trainable(model, targets, target_classes, trainable=True)

def count_params(model: nn.Module, mode: str = 'trainable') -> ParamStats:
    '''统计模型参数数量并获取参数列表。

    Args:
        model (nn.Module): 目标模型。
        mode (str): 统计模式，可选 'trainable', 'frozen', 'all'。默认为 'trainable'。

    Returns:
        ParamStats: 包含参数总数(count)和参数列表(params)的数据类。
    '''
    if mode == 'trainable':
        params = [p for p in model.parameters() if p.requires_grad]
    elif mode == 'frozen':
        params = [p for p in model.parameters() if not p.requires_grad]
    elif mode == 'all':
        params = list(model.parameters())
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be one of 'trainable', 'frozen', 'all'.")
    
    count = sum(p.numel() for p in params)
    return ParamStats(count=count, params=params)
