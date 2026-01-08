import json
import os
from typing import Any, Dict

class BaseConfig:
    '''基础配置类，用于管理模型超参数。
    
    支持从 JSON 文件加载和保存，以及字典风格的属性访问。
    '''
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_pretrained(cls, path: str) -> 'BaseConfig':
        '''从 JSON 文件加载配置。
        
        Args:
            path (str): JSON 文件路径。
            
        Returns:
            BaseConfig: 加载的配置对象。
        '''
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
            
        return cls(**config_dict)

    def save_pretrained(self, path: str):
        '''将配置保存到 JSON 文件。
        
        Args:
            path (str): 保存路径。
        '''
        # 确保目录存在
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)

    def to_dict(self) -> Dict[str, Any]:
        '''转换为字典。'''
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.to_dict()})"
