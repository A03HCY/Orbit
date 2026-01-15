import torch
import torch.nn as nn

from typing import Optional
from dataclasses import dataclass

from orbit.model import BaseBlock, register_model

@dataclass
class FiLMOutput:
    ''' FiLM 模块的输出容器。
    
    Attributes:
        x (torch.Tensor): 经过 gamma 和 beta 调制后的特征。
        gate (Optional[torch.Tensor]): 用于残差连接的门控值。
    '''
    x: torch.Tensor
    gate: Optional[torch.Tensor] = None


@register_model()
class FiLM(BaseBlock):
    ''' Feature-wise Linear Modulation (FiLM) 模块。

    对输入特征进行仿射变换：FiLM(x) = (1 + gamma(z)) * x + beta(z)
    其中 gamma 和 beta 是从条件输入 z 生成的。
    初始状态下，gamma 为 0，beta 为 0，即恒等映射。

    Args:
        in_features (int): 输入特征维度。
        cond_features (int): 条件特征维度。
        use_beta (bool, optional): 是否使用平移项 (beta)。默认为 True。
        use_gamma (bool, optional): 是否使用缩放项 (gamma)。默认为 True。
        use_gate (bool, optional): 是否使用门控项 (gate)。默认为 True。
        channel_first (bool, optional): 特征维度是否在第 1 维 (如 CNN [B, C, H, W])。
            如果为 False，则假设特征在最后一维 (如 Transformer [B, L, C])。默认为 False。
    '''
    def __init__(
        self,
        in_features: int,
        cond_features: int,
        use_beta: bool = True,
        use_gamma: bool = True,
        use_gate: bool = True,
        channel_first: bool = False
    ):
        super(FiLM, self).__init__()
        self.in_features = in_features
        self.cond_features = cond_features
        self.use_beta = use_beta
        self.use_gamma = use_gamma
        self.use_gate = use_gate
        self.channel_first = channel_first

        self.out_dim = 0
        if use_gamma: self.out_dim += in_features
        if use_beta:  self.out_dim += in_features
        if use_gate:  self.out_dim += in_features

        if self.out_dim > 0:
            self.proj = nn.Linear(cond_features, self.out_dim)
            nn.init.constant_(self.proj.weight, 0)
            nn.init.constant_(self.proj.bias, 0)
        else: self.proj = None
    
    def _init_weights(self, model: nn.Module):
        ''' 初始化权重。

        将投影层的权重和偏置初始化为 0，以确保初始状态为恒等映射。

        Args:
            model (nn.Module): 需要初始化的模型。
        '''
        if model is self:
            nn.init.constant_(self.proj.weight, 0)
            nn.init.constant_(self.proj.bias, 0)
            return

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> FiLMOutput:
        ''' 前向传播。

        Args:
            x (torch.Tensor): 输入特征。形状为 [B, C, ...] (如果 channel_first=True)
                或 [B, ..., C] (如果 channel_first=False)。
            cond (torch.Tensor): 条件输入。形状为 [B, cond_features]。

        Returns:
            FiLMOutput: 调制后的特征。
        '''
        if self.proj is None: return FiLMOutput(x=x)
        
        params = self.proj(cond)
        
        count = sum([self.use_gamma, self.use_beta, self.use_gate])
        if count > 1:
            params_list = params.chunk(count, dim=-1)
        else:
            params_list = [params]
        
        idx = 0
        gamma, beta, gate = None, None, None
        if self.use_gamma:
            gamma = params_list[idx]
            idx += 1
        if self.use_beta:
            beta = params_list[idx]
            idx += 1
        if self.use_gate:
            gate = params_list[idx]
            idx += 1
            
        ndim = x.ndim
        if self.channel_first:
            shape = [x.shape[0], self.in_features] + [1] * (ndim - 2)
        else:
            shape = [x.shape[0]] + [1] * (ndim - 2) + [self.in_features]
        
        out = x
        if gamma is not None:
            out = out * (1 + gamma.view(*shape))
        if beta is not None:
            out = out + beta.view(*shape)
            
        final_gate = None
        if gate is not None:
            final_gate = gate.view(*shape)
        return FiLMOutput(x=out, gate=final_gate)
