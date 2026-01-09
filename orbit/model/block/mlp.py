import torch
import torch.nn as nn
import torch.nn.functional as F

from orbit.model import BaseBlock, register_model


@register_model()
class MLP(BaseBlock):
    ''' 多层感知机 (MLP) 模块。

    支持标准 MLP 和门控 MLP (Gated MLP) 结构。

    Args:
        in_features (int): 输入特征维度。
        hidden_features (int): 隐藏层特征维度。
        out_features (int, optional): 输出特征维度。如果为 None，则等于 in_features。默认为 None。
        gate (bool, optional): 是否使用门控机制。默认为 False。
        dropout (float, optional): Dropout 概率。默认为 0.0。
    '''
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int = None,
        use_gate: bool = False,
        dropout: float = 0.0
    ):
        super(MLP, self).__init__()
        
        out_features = out_features or in_features

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.use_gate = use_gate
        self.dropout = nn.Dropout(dropout)
        
        if use_gate:
            self.gate_proj = nn.Linear(in_features, hidden_features)
            self.up_proj = nn.Linear(in_features, hidden_features)
            self.down_proj = nn.Linear(hidden_features, out_features)
            self.act = nn.SiLU()
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.act = nn.GELU()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' 前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        '''
        if self.use_gate:
            return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
        else:
            x = self.fc1(x)
            x = self.act(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.dropout(x)
            return x
