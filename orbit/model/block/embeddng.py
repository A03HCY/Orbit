import torch
import torch.nn as nn
import math

from orbit.model import BaseBlock, register_model


@register_model()
class RotaryPositionalEmbedding(BaseBlock):
    '''
    旋转位置编码 (Rotary Positional Embedding, RoPE)。
    '''

    def __init__(self, model_dim: int, max_len: int = 128000, base: int = 10000):
        '''
        初始化 RoPE 模块。

        Args:
            model_dim (int): 模型的维度 (或 head_dim)。必须是偶数。
            max_len (int, optional): 预计算位置编码的最大序列长度。默认为 128000。
            base (int, optional): 计算频率的基数。默认为 10000。
        '''
        super(RotaryPositionalEmbedding, self).__init__()

        self.model_dim = model_dim
        self.max_len = max_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, model_dim, 2).float() / model_dim))
        
        t = torch.arange(max_len, dtype=torch.float)
        
        freqs = torch.outer(t, inv_freq)
        
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        '''
        将向量分为两半并旋转: [-x2, x1]。
        无论输入是 3D 还是 4D，Split 都是作用在最后一维 (model_dim)。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 旋转后的张量。
        '''
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        '''
        应用旋转位置编码。
        
        自动适配两种输入:
        1. [Batch, Seq_Len, Dim]
        2. [Batch, Head, Seq_Len, Head_Dim]

        Args:
            x (torch.Tensor): 输入张量。
            start_pos (int, optional): 起始位置索引，用于 KV Cache 推理。默认为 0。

        Returns:
            torch.Tensor: 添加了位置信息的张量。
        '''
        ndim = x.ndim
        seq_len = x.shape[-2]

        cos = self.cos_cached[start_pos : start_pos + seq_len, :]
        sin = self.sin_cached[start_pos : start_pos + seq_len, :]
        
        shape = [1] * (ndim - 2) + [seq_len, -1]
        cos = cos.view(*shape)
        sin = sin.view(*shape)

        return (x * cos) + (self._rotate_half(x) * sin)


@register_model()
class SinusoidalPositionalEmbedding(BaseBlock):

    def __init__(self, model_dim: int, max_len: int = 128000):
        '''
        初始化绝对位置编码模块。

        Args:
            model_dim (int): 模型的维度。
            max_len (int, optional): 最大序列长度。默认为 128000。
        '''
        super(SinusoidalPositionalEmbedding, self).__init__()

        self.model_dim = model_dim
        self.max_len = max_len
        
        pe = torch.zeros(max_len, model_dim)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        前向传播。

        Args:
            x (torch.Tensor): 输入张量。Shape: [Batch_Size, Seq_Len, model_dim]。

        Returns:
            torch.Tensor: 加上位置编码后的张量。
        '''
        x = x + self.pe[:, :x.size(1), :]
        return x

@register_model()
class MRoPEInterleavedEmbedding(BaseBlock):
    '''
    交错分配多模态旋转位置编码 (MRoPE‑Interleave)。
    支持三维位置（时间 t、高度 h、宽度 w），频率通道采用轮转交错分配 (thw…thw…thw)。
    '''
    def __init__(self, model_dim: int, max_len: int = 128000, base: int = 10000, num_axes: int = 3):
        '''
        初始化 MRoPEInterleaved 模块。

        Args:
            model_dim (int): 模型的维度。必须是偶数且能被 num_axes 整除。
            max_len (int, optional): 预计算位置编码的最大序列长度。默认为 128000。
            base (int, optional): 计算频率的基数。默认为 10000。
            num_axes (int, optional): 位置轴的数量（例如 3 表示时间、高度、宽度）。默认为 3。
        '''
        super().__init__()
        assert model_dim % 2 == 0, 'model_dim must be even'
        assert model_dim % num_axes == 0, f'model_dim {model_dim} not divisible by num_axes {num_axes}'
        
        self.model_dim = model_dim
        self.max_len = max_len
        self.base = base
        self.num_axes = num_axes
        
        inv_freq = 1.0 / (base ** (torch.arange(0, model_dim, 2).float() / model_dim))
        
        t_range = torch.arange(max_len, dtype=torch.float)
        freqs = torch.outer(t_range, inv_freq)  # [max_len, dim/2]
        
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_len, dim]
        
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
        
        self.register_buffer(
            'axis_mask', 
            torch.arange(model_dim) % num_axes, 
            persistent=False
        )
        
        k = model_dim // num_axes
        idx = []
        for p in range(model_dim):
            j = p % num_axes
            i = p // num_axes
            pos_in_old = j * k + i
            idx.append(pos_in_old)
            
        self.register_buffer('interleave_idx', torch.tensor(idx, dtype=torch.long), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        '''
        将向量分为两半并旋转: [-x2, x1]。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 旋转后的张量。
        '''
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, positions: torch.Tensor = None, start_pos: int = 0) -> torch.Tensor:
        '''
        应用多模态旋转位置编码。

        Args:
            x (torch.Tensor): 输入张量。Shape: [Batch, Seq_Len, Dim] 或 [Batch, Head, Seq_Len, Head_Dim]。
            positions (torch.Tensor, optional): 位置索引张量。Shape: [Batch, Seq_Len] 或 [Batch, Seq_Len, num_axes]。
                如果是 2D 张量，将自动扩展为 [Batch, Seq_Len, num_axes]。
                如果为 None 且 num_axes=1，将自动创建线性位置索引。
            start_pos (int, optional): 起始位置索引。默认为 0。

        Returns:
            torch.Tensor: 添加了位置信息的张量。
        
        Raises:
            ValueError: 如果 positions 为 None 且 num_axes > 1。
        '''
        ndim = x.ndim
        seq_len = x.shape[-2]
        batch_size = x.shape[0]
        
        if positions is None:
            if self.num_axes == 1:
                positions = torch.arange(0, seq_len, device=x.device, dtype=torch.long)
            else:
                raise ValueError("positions must be provided when num_axes > 1 (e.g. for vision/multimodal inputs)")
        
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, self.num_axes)
            
        if positions.ndim == 2:
            positions = positions.unsqueeze(-1).expand(-1, -1, self.num_axes)
            
        if positions.ndim == 3 and positions.shape[-1] == 1:
            positions = positions.expand(-1, -1, self.num_axes)
            
        batch_size = positions.shape[0]
        
        cos_list, sin_list = [], []
        
        for ax in range(self.num_axes):
            pos_ax = positions[..., ax]
            pos_ax = torch.clamp(pos_ax + start_pos, 0, self.max_len - 1).long()
            
            cos_full = self.cos_cached[pos_ax]
            sin_full = self.sin_cached[pos_ax]
            
            mask = (self.axis_mask == ax)
            cos_ax = cos_full[..., mask]
            sin_ax = sin_full[..., mask]
            
            cos_list.append(cos_ax)
            sin_list.append(sin_ax)
        
        cos_all = torch.cat(cos_list, dim=-1)
        sin_all = torch.cat(sin_list, dim=-1)
        
        cos_all = cos_all[..., self.interleave_idx]
        sin_all = sin_all[..., self.interleave_idx]
        
        if ndim == 4:
            shape = [batch_size, 1, seq_len, -1]
            cos_all = cos_all.view(*shape)
            sin_all = sin_all.view(*shape)
        
        return (x * cos_all) + (self._rotate_half(x) * sin_all)
