import sys
import inspect
from typing import Dict, Type, Any, Optional

_MODEL_REGISTRY: Dict[str, Type] = {}

def register_model(name: Optional[str] = None):
    '''模型注册装饰器。
    
    Args:
        name (str, optional): 模型名称。如果未提供，则使用类名。
    '''
    def decorator(cls):
        model_name = name if name is not None else cls.__name__
        if model_name in _MODEL_REGISTRY:
            print(f"Warning: Model '{model_name}' is already registered. Overwriting.")
        _MODEL_REGISTRY[model_name] = cls
        return cls
    return decorator

def build_model(name: str, **kwargs) -> Any:
    '''根据名称构建模型。
    
    Args:
        name (str): 模型名称。
        **kwargs: 传递给模型构造函数的参数。
        
    Returns:
        Any: 实例化的模型对象。
        
    Raises:
        ValueError: 如果模型名称未注册。
    '''
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry. Available models: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name](**kwargs)

def list_models() -> list:
    '''列出所有已注册的模型名称。'''
    return list(_MODEL_REGISTRY.keys())

def get_model_class(name: str) -> Type:
    '''获取模型类。
    
    Args:
        name (str): 模型名称。
        
    Returns:
        Type: 模型类。
    '''
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry.")
    return _MODEL_REGISTRY[name]
