import torch
import torch.nn as nn

from orbit.model import BaseBlock, register_model


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
        channel_first (bool, optional): 特征维度是否在第 1 维 (如 CNN [B, C, H, W])。
            如果为 False，则假设特征在最后一维 (如 Transformer [B, L, C])。默认为 False。
    '''
    def __init__(
        self,
        in_features: int,
        cond_features: int,
        use_beta: bool = True,
        use_gamma: bool = True,
        channel_first: bool = False
    ):
        super(FiLM, self).__init__()
        self.in_features = in_features
        self.cond_features = cond_features
        self.use_beta = use_beta
        self.use_gamma = use_gamma
        self.channel_first = channel_first

        self.out_dim = 0
        if use_gamma: self.out_dim += in_features
        if use_beta: self.out_dim += in_features

        if self.out_dim > 0:
            self.proj = nn.Linear(cond_features, self.out_dim)
            nn.init.constant_(self.proj.weight, 0)
            nn.init.constant_(self.proj.bias, 0)
        else:
            self.proj = None
    
    def _init_weights(self, model: nn.Module):
        if model is self:
            nn.init.constant_(self.proj.weight, 0)
            nn.init.constant_(self.proj.bias, 0)
            return

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        ''' 前向传播。

        Args:
            x (torch.Tensor): 输入特征。
            cond (torch.Tensor): 条件输入。

        Returns:
            torch.Tensor: 调制后的特征。
        '''
        if self.proj is None:
            return x

        params = self.proj(cond)

        gamma, beta = None, None
        if self.use_gamma and self.use_beta:
            gamma, beta = params.chunk(2, dim=-1)
        elif self.use_gamma:
            gamma = params
        elif self.use_beta:
            beta = params

        ndim = x.ndim
        if self.channel_first:
            # [B, C] -> [B, C, 1, 1, ...]
            shape = [x.shape[0], self.in_features] + [1] * (ndim - 2)
        else:
            # [B, C] -> [B, 1, 1, ..., C]
            shape = [x.shape[0]] + [1] * (ndim - 2) + [self.in_features]

        out = x
        if gamma is not None:
            gamma = gamma.view(*shape)
            out = out * (1 + gamma)
        
        if beta is not None:
            beta = beta.view(*shape)
            out = out + beta
            
        return out
