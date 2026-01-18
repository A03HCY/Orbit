import torch
import torch.nn as nn

from typing import Optional, List
from dataclasses import dataclass

from orbit.model import BaseBlock, register_model


@register_model()
class LowRankFusion(BaseBlock):
    ''' Low-rank Multimodal Fusion (LMF) 模块。
    
    使用低秩分解近似多模态外积，通过将权重张量分解为模态特定因子的组合来实现高效融合。
    '''

    def __init__(self, in_features: List[int], out_features: int, rank: int, dropout: float = 0.0, channel_first: bool = False):
        ''' 初始化 LowRankFusion。
        
        Args:
            in_features (List[int]): 每个输入模态的特征维度列表。
            out_features (int): 融合后的输出特征维度。
            rank (int): 低秩分解的秩（rank）。
            dropout (float, optional): Dropout 概率。默认为 0.0。
            channel_first (bool, optional): 特征维度是否在第 1 维 (如 CNN [B, C, H, W])。
                如果为 False，则假设特征在最后一维 (如 Transformer [B, L, C])。默认为 False。
        '''
        super().__init__()
        
        if not in_features:
            raise ValueError("in_features cannot be empty")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.channel_first = channel_first
        
        self.modality_factors = nn.ModuleList([
            nn.Linear(dim, rank, bias=True) for dim in in_features
        ])
        
        self.fusion_weights = nn.Linear(rank, out_features, bias=True)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights(self)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        ''' 前向传播。
        
        Args:
            inputs (List[torch.Tensor]): 输入张量列表。所有张量在除最后一个维度外的其他维度应匹配。
            
        Returns:
            torch.Tensor: 融合后的输出张量。
        '''
        if len(inputs) != len(self.in_features):
            raise ValueError(f"Expected {len(self.in_features)} inputs, got {len(inputs)}")
        
        if self.channel_first:
            inputs = [x.movedim(1, -1) for x in inputs]
            
        fusion_tensor: Optional[torch.Tensor] = None
        
        for i, x in enumerate(inputs):
            projected = self.modality_factors[i](x)
            
            if fusion_tensor is None:
                fusion_tensor = projected
            else:
                fusion_tensor = fusion_tensor * projected
        
        if fusion_tensor is None:
             raise ValueError("No inputs processed")

        output = self.fusion_weights(self.dropout(fusion_tensor))
        
        if self.channel_first:
            output = output.movedim(-1, 1)
        
        return output


@register_model()
class GatedMultimodalUnit(BaseBlock):
    ''' Gated Multimodal Unit (GMU) 模块。
    
    通过学习门控机制来控制每个模态对最终融合表示的贡献。
    '''

    def __init__(self, in_features: List[int], out_features: int, channel_first: bool = False):
        ''' 初始化 GatedMultimodalUnit。
        
        Args:
            in_features (List[int]): 每个输入模态的特征维度列表。
            out_features (int): 隐藏层特征维度。
            channel_first (bool, optional): 特征维度是否在第 1 维。默认为 False。
        '''
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.channel_first = channel_first
        
        self.feature_transforms = nn.ModuleList([
            nn.Linear(dim, out_features) for dim in in_features
        ])
        
        total_in_features = sum(in_features)
        self.gate_net = nn.Linear(total_in_features, len(in_features))
        
        self._init_weights(self)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        ''' 前向传播。
        
        Args:
            inputs (List[torch.Tensor]): 输入张量列表。
            
        Returns:
            torch.Tensor: 融合后的输出张量。
        '''
        if len(inputs) != len(self.in_features):
            raise ValueError(f"Expected {len(self.in_features)} inputs, got {len(inputs)}")
        
        processed_inputs = []
        if self.channel_first:
            processed_inputs = [x.movedim(1, -1) for x in inputs]
        else:
            processed_inputs = inputs
            
        hidden_features = []
        for i, x in enumerate(processed_inputs):
            h = torch.tanh(self.feature_transforms[i](x))
            hidden_features.append(h)
            
        concatenated_input = torch.cat(processed_inputs, dim=-1)
        gate_logits = self.gate_net(concatenated_input) # (B, ..., num_modalities)
        gates = torch.softmax(gate_logits, dim=-1)
        
        output = torch.zeros_like(hidden_features[0])
        
        for i, h in enumerate(hidden_features):
            g = gates[..., i:i+1] # (B, ..., 1)
            output += g * h
            
        if self.channel_first:
            output = output.movedim(-1, 1)
            
        return output


@register_model()
class DiffusionMapsFusion(BaseBlock):
    ''' Diffusion Maps Fusion 模块。
    
    基于流形学习中的扩散映射思想。
    通过在特征通道间构建图拉普拉斯算子（或归一化亲和矩阵），
    在流形空间进行特征对齐和交叉扩散 (Cross-diffusion)。
    
    目前主要支持两个模态的融合。
    '''

    def __init__(self, in_features: List[int], out_features: int, sigma: float = 1.0, channel_first: bool = False):
        ''' 初始化 DiffusionMapsFusion。
        
        Args:
            in_features (List[int]): 两个输入模态的特征维度列表。
            out_features (int): 输出特征维度。
            sigma (float, optional): 高斯核的带宽参数。默认为 1.0。
            channel_first (bool, optional): 特征维度是否在第 1 维。默认为 False。
        '''
        super().__init__()
        
        if len(in_features) != 2:
            raise ValueError("DiffusionMapsFusion currently only supports exactly 2 modalities.")
            
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma
        self.channel_first = channel_first
        
        self.projections = nn.ModuleList([
            nn.Linear(dim, out_features) for dim in in_features
        ])
        
        self.output_proj = nn.Linear(out_features * 2, out_features)
        
        self._init_weights(self)
        
    def _compute_affinity(self, x: torch.Tensor) -> torch.Tensor:
        ''' 计算特征通道间的归一化亲和矩阵 (Diffusion Operator)。
        
        Args:
            x (torch.Tensor): 输入特征 (C, N)。C 是特征通道数，N 是样本数。
            
        Returns:
            torch.Tensor: 归一化的亲和矩阵 (C, C)。
        '''
        # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 <x_i, x_j>
        sq_norm = (x ** 2).sum(1, keepdim=True)
        dist_sq = sq_norm + sq_norm.t() - 2 * torch.mm(x, x.t())
        
        W = torch.exp(-dist_sq / (2 * self.sigma ** 2))
        
        D_inv = 1.0 / (W.sum(1, keepdim=True) + 1e-8)
        P = D_inv * W
        
        return P

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        ''' 前向传播。
        
        Args:
            inputs (List[torch.Tensor]): 两个输入张量列表。
            
        Returns:
            torch.Tensor: 融合后的输出张量。
        '''
        if len(inputs) != 2:
            raise ValueError("Expected exactly 2 inputs")
            
        processed_inputs = []
        if self.channel_first:
            processed_inputs = [x.movedim(1, -1) for x in inputs]
        else:
            processed_inputs = inputs
            
        proj_feats = [self.projections[i](x) for i, x in enumerate(processed_inputs)]
        xA, xB = proj_feats[0], proj_feats[1]
        
        flat_xA = xA.reshape(-1, xA.shape[-1])
        flat_xB = xB.reshape(-1, xB.shape[-1])
        
        xA_T = flat_xA.t()
        xB_T = flat_xB.t()
        
        P_A = self._compute_affinity(xA_T) # (C, C)
        P_B = self._compute_affinity(xB_T) # (C, C)
        
        diffused_A_T = torch.mm(P_B, xA_T) # (C, N_samples)
        diffused_B_T = torch.mm(P_A, xB_T) # (C, N_samples)
        
        diffused_A = diffused_A_T.t().view(xA.shape)
        diffused_B = diffused_B_T.t().view(xB.shape)
        
        combined = torch.cat([diffused_A, diffused_B], dim=-1)
        output = self.output_proj(combined)
        
        if self.channel_first:
            output = output.movedim(-1, 1)
            
        return output


@register_model()
class CompactMultimodalPooling(BaseBlock):
    ''' Compact Multimodal Pooling (MCB/CBP) 模块。
    
    通过 Count Sketch 和 FFT 近似多模态特征的外积。
    支持两个或多个模态的融合。
    '''

    def __init__(self, in_features: List[int], out_features: int, channel_first: bool = False):
        ''' 初始化 CompactMultimodalPooling。
        
        Args:
            in_features (List[int]): 每个输入模态的特征维度列表。
            out_features (int): 输出特征维度。通常应该比输入维度高，以保持信息。
            channel_first (bool, optional): 特征维度是否在第 1 维 (如 CNN [B, C, H, W])。
                如果为 False，则假设特征在最后一维 (如 Transformer [B, L, C])。默认为 False。
        '''
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.channel_first = channel_first
        
        for i, dim in enumerate(in_features):
            self.register_buffer(f'h_{i}', torch.randint(0, out_features, (dim,)))
            self.register_buffer(f's_{i}', torch.randint(0, 2, (dim,)) * 2 - 1) # Map {0, 1} to {-1, 1}

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        ''' 前向传播。
        
        Args:
            inputs (List[torch.Tensor]): 输入张量列表。
            
        Returns:
            torch.Tensor: 融合后的输出张量。
        '''
        if len(inputs) != len(self.in_features):
            raise ValueError(f"Expected {len(self.in_features)} inputs, got {len(inputs)}")
        
        if self.channel_first:
            inputs = [x.movedim(1, -1) for x in inputs]
            
        batch_size = inputs[0].size(0)
        fft_product: Optional[torch.Tensor] = None
        
        for i, x in enumerate(inputs):
            h = getattr(self, f'h_{i}') # (dim,)
            s = getattr(self, f's_{i}') # (dim,)
            
            output_shape = list(x.shape)
            output_shape[-1] = self.out_features
            sketch = torch.zeros(output_shape, device=x.device, dtype=x.dtype)
            
            weighted_x = x * s # (..., dim)
            
            flat_x = weighted_x.reshape(-1, weighted_x.shape[-1]) # (N, dim)
            flat_sketch = sketch.view(-1, self.out_features) # (N, out)
            
            h_expanded = h.expand(flat_x.shape[0], -1)
            
            flat_sketch.scatter_add_(1, h_expanded, flat_x)
            
            sketch = flat_sketch.view(output_shape)
            
            fft_x = torch.fft.rfft(sketch, dim=-1)
            
            if fft_product is None:
                fft_product = fft_x
            else:
                fft_product = fft_product * fft_x
        
        if fft_product is None:
            raise ValueError("No inputs processed")
            
        output = torch.fft.irfft(fft_product, n=self.out_features, dim=-1)
        
        if self.channel_first:
            output = output.movedim(-1, 1)
        
        return output
