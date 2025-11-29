import torch
import torch.nn as nn
from typing import Union, List, Optional, Iterable

def set_trainable(
    model: nn.Module, 
    targets: Optional[Union[str, List[str]]] = None, 
    trainable: bool = False
) -> None:
    '''设置模型参数的 requires_grad 属性，用于冻结或解冻层。

    Args:
        model (nn.Module): 目标模型。
        targets (str or List[str], optional): 要操作的层名称或参数名称模式。
            - 如果为 None，则操作模型的所有参数。
            - 如果为 str，则操作名称中包含该字符串的所有参数。
            - 如果为 List[str]，则操作名称中包含列表中任意字符串的所有参数。
        trainable (bool): 是否可训练 (True 为解冻, False 为冻结)。
    '''
    if targets is None:
        for param in model.parameters():
            param.requires_grad = trainable
    else:
        if isinstance(targets, str):
            targets = [targets]
        
        for name, param in model.named_parameters():
            # 检查参数名是否包含 targets 中的任何一个模式
            if any(t in name for t in targets):
                param.requires_grad = trainable

def freeze_layers(model: nn.Module, targets: Optional[Union[str, List[str]]] = None) -> None:
    '''冻结模型指定层或所有层 (requires_grad=False)。

    Args:
        model (nn.Module): 目标模型。
        targets (str or List[str], optional): 要冻结的层名称模式。如果不指定，则冻结整个模型。
    '''
    set_trainable(model, targets, trainable=False)

def unfreeze_layers(model: nn.Module, targets: Optional[Union[str, List[str]]] = None) -> None:
    '''解冻模型指定层或所有层 (requires_grad=True)。

    Args:
        model (nn.Module): 目标模型。
        targets (str or List[str], optional): 要解冻的层名称模式。如果不指定，则解冻整个模型。
    '''
    set_trainable(model, targets, trainable=True)

def get_trainable_params(model: nn.Module) -> Iterable[torch.Tensor]:
    '''获取模型中 requires_grad=True 的参数，供优化器使用。

    Args:
        model (nn.Module): 目标模型。

    Returns:
        Iterable[torch.Tensor]: 可训练参数的迭代器。
    '''
    return filter(lambda p: p.requires_grad, model.parameters())
