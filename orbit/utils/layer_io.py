import torch
import torch.nn as nn
from typing import Union

def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    '''通过名称获取模型的子模块。
    
    Args:
        model (nn.Module): 目标模型。
        name (str): 子模块名称，例如 'backbone.layer1'。
        
    Returns:
        nn.Module: 找到的子模块。
        
    Raises:
        AttributeError: 如果找不到指定的模块名称。
    '''
    names = name.split('.')
    module = model
    for n in names:
        if not hasattr(module, n):
            raise AttributeError(f"Module '{type(module).__name__}' has no attribute '{n}'")
        module = getattr(module, n)
    return module

def save_layer_weights(model: nn.Module, layer_name: str, file_path: str) -> None:
    '''保存模型指定层的权重到文件。
    
    Args:
        model (nn.Module): 目标模型。
        layer_name (str): 要保存权重的层名称。
        file_path (str): 保存路径。
    '''
    module = get_module_by_name(model, layer_name)
    torch.save(module.state_dict(), file_path)

def load_layer_weights(
    model: nn.Module, 
    layer_name: str, 
    file_path: str, 
    strict: bool = True,
    map_location: Union[str, torch.device] = 'cpu'
) -> None:
    '''从文件加载权重到模型的指定层。
    
    Args:
        model (nn.Module): 目标模型。
        layer_name (str): 要加载权重的层名称。
        file_path (str): 权重文件路径。
        strict (bool): 是否严格匹配键值。默认为 True。
        map_location (str or torch.device): 加载位置。默认为 'cpu'。
    '''
    state_dict = torch.load(file_path, map_location=map_location)
    module = get_module_by_name(model, layer_name)
    module.load_state_dict(state_dict, strict=strict)
