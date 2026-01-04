import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class LinearLoRA(nn.Module):
    '''实现 Linear 层的 LoRA (Low-Rank Adaptation)。

    LoRA 通过注入可训练的低秩矩阵来适应预训练权重，同时冻结原始权重。
    计算公式: h = W_0 x + B A x * scaling

    Attributes:
        original_layer (nn.Linear): 原始的 Linear 层。
        r (int): LoRA 的秩。
        lora_alpha (int): LoRA 的缩放系数。
        scaling (float): 实际缩放比例 (lora_alpha / r)。
        gate (bool): 是否使用 Gated LoRA。
        lora_gate (nn.Parameter): 门控参数。
        dora (bool): 是否使用 DoRA。
        dora_m (nn.Parameter): DoRA 的幅值向量。
        merged (bool): 权重是否已合并。
        lora_a (nn.Parameter): 降维矩阵 A。
        lora_b (nn.Parameter): 升维矩阵 B。
    '''
    def __init__(
        self, 
        original_layer: nn.Linear, 
        r: int = 8, 
        lora_alpha: int = 16, 
        lora_dropout: float = 0.05,
        merge_weights: bool = False,
        gate: bool = False,
        dora: bool = False
    ):
        '''初始化 LinearLoRA。

        Args:
            original_layer (nn.Linear): 原始的 Linear 层。
            r (int): LoRA 的秩。默认为 8。
            lora_alpha (int): LoRA 的缩放系数。默认为 16。
            lora_dropout (float): Dropout 概率。默认为 0.05。
            merge_weights (bool): 初始化时是否将 LoRA 权重合并到原始权重中。默认为 False。
            gate (bool): 是否使用 Gated LoRA。默认为 False。
            dora (bool): 是否使用 DoRA。默认为 False。
        '''
        super(LinearLoRA, self).__init__()
        
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        self.original_layer = original_layer
        for p in self.original_layer.parameters():
            p.requires_grad = False
            
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.merged = False
        self.gate = gate
        
        if r > 0:
            self.lora_gate = nn.Parameter(torch.tensor([1.0])) if gate else None
            self.lora_a = nn.Parameter(torch.zeros((r, self.in_features)))
            self.lora_b = nn.Parameter(torch.zeros((self.out_features, r)))
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_a = None
            self.lora_b = None
            self.lora_dropout = None
            
        self.reset_parameters()
        
        self.dora = dora
        if dora and r > 0:
            self.dora_m = nn.Parameter(self.original_layer.weight.norm(p=2, dim=0, keepdim=True))
        else:
            self.dora_m = None

        if merge_weights: self.merge()

    def reset_parameters(self):
        '''重置 LoRA 参数。
        
        A 矩阵使用 Kaiming Uniform 初始化，B 矩阵初始化为零。
        这样可以确保初始状态下 LoRA 分支的输出为零，不影响模型原有输出。
        '''
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b)

    def train(self, mode: bool = True):
        '''设置训练模式。

        如果进入训练模式，确保权重未合并。
        
        Args:
            mode (bool): 是否为训练模式。
        '''
        super().train(mode)
        if not mode: return
        if self.merged: self.unmerge()

    def merge(self):
        '''将 LoRA 权重合并到原始层权重中。
        
        用于推理加速。
        DoRA: 合并后无法恢复原始权重（除非存储原始权重副本，但这违背了LoRA节省显存的初衷）。
        '''
        if self.r > 0 and not self.merged:
            if self.dora:
                # Calculate full weight W' = W0 + BA * scaling
                delta_w = (self.lora_b @ self.lora_a) * self.scaling
                if self.gate: delta_w *= self.lora_gate
                weight = self.original_layer.weight + delta_w
                
                # Normalize and scale: W_final = m * W' / ||W'||
                norm = weight.norm(p=2, dim=1, keepdim=True)
                weight = (weight / (norm + 1e-6)) * self.dora_m
                
                # Update original weight (Destructive!)
                self.original_layer.weight.data = weight
            else:
                # W_new = W_old + B @ A * scaling
                delta_w = (self.lora_b @ self.lora_a) * self.scaling
                if self.gate: delta_w *= self.lora_gate
                self.original_layer.weight.data += delta_w
            
            self.merged = True

    def unmerge(self):
        '''从原始权重中减去 LoRA 权重。
        
        用于恢复原始权重或继续训练。
        注意：DoRA 模式下不支持 unmerge。
        '''
        if self.r > 0 and self.merged:
            if self.dora:
                print("Warning: DoRA weights cannot be unmerged exactly. Original weights are lost.")
                # We can't easily recover W0 from W_final = m * (W0 + BA) / ||...||
                # So we just set merged to False and hope the user knows what they are doing (e.g. for training continuation, but W0 is now modified)
                # Ideally, one should save initial W0 if unmerge is required.
                pass 
            else:
                delta_w = (self.lora_b @ self.lora_a) * self.scaling
                if self.gate: delta_w *= self.lora_gate
                self.original_layer.weight.data -= delta_w
            
            self.merged = False

    def forward(self, x: torch.Tensor):
        '''前向传播。
        
        Args:
            x (torch.Tensor): 输入张量。
            
        Returns:
            torch.Tensor: 输出张量。
        '''
        if self.r > 0 and self.merged:
            return self.original_layer(x)
        
        if self.dora and self.r > 0:
            # DoRA: W_final = m * (W0 + BA) / ||W0 + BA||
            delta_w = (self.lora_b @ self.lora_a) * self.scaling
            if self.gate: delta_w *= self.lora_gate
            
            # Reconstruct full weight for calculation
            weight = self.original_layer.weight + delta_w
            norm = weight.norm(p=2, dim=1, keepdim=True)
            weight = (weight / (norm + 1e-6)) * self.dora_m
            
            return F.linear(x, weight, self.original_layer.bias)
        
        result = self.original_layer(x)
        
        if self.r > 0:
            # x shape: (batch, ..., in)
            # lora_a shape: (r, in) -> x @ A.T -> (batch, ..., r)
            # lora_b shape: (out, r) -> result @ B.T -> (batch, ..., out)
            x_dropped = self.lora_dropout(x)
            lora_out = (x_dropped @ self.lora_a.transpose(0, 1) @ self.lora_b.transpose(0, 1)) * self.scaling
            if self.gate: lora_out *= self.lora_gate
            result += lora_out
            
        return result

    def __repr__(self):
        prefix = 'Gated' if self.gate else ''
        suffix = 'DoRA' if self.dora else 'LoRA'
        return f'{self.__class__.__name__}(type={prefix}{suffix}, in_features={self.in_features}, out_features={self.out_features}, r={self.r}, merged={self.merged})'

class Conv2dLoRA(nn.Module):
    '''实现 Conv2d 层的 LoRA (Low-Rank Adaptation)。

    使用两个连续的卷积层模拟低秩矩阵分解：
    1. A 层: 降低通道数到 r，保持 kernel_size。
    2. B 层: 恢复通道数，使用 1x1 kernel。

    Attributes:
        original_layer (nn.Conv2d): 原始的 Conv2d 层。
        r (int): LoRA 的秩。
        lora_alpha (int): LoRA 的缩放系数。
        scaling (float): 实际缩放比例 (lora_alpha / r)。
        gate (bool): 是否使用 Gated LoRA。
        lora_gate (nn.Parameter): 门控参数。
        dora (bool): 是否使用 DoRA。
        dora_m (nn.Parameter): DoRA 的幅值向量。
        merged (bool): 权重是否已合并。
        lora_a (nn.Conv2d): 降维卷积层。
        lora_b (nn.Conv2d): 升维卷积层 (1x1)。
    '''
    def __init__(
        self, 
        original_layer: nn.Conv2d, 
        r: int = 8, 
        lora_alpha: int = 16, 
        lora_dropout: float = 0.05,
        merge_weights: bool = False,
        gate: bool = False,
        dora: bool = False
    ):
        '''初始化 Conv2dLoRA。

        Args:
            original_layer (nn.Conv2d): 原始的 Conv2d 层。
            r (int): LoRA 的秩。默认为 8。
            lora_alpha (int): LoRA 的缩放系数。默认为 16。
            lora_dropout (float): Dropout 概率。默认为 0.05。
            merge_weights (bool): 初始化时是否将 LoRA 权重合并到原始权重中。默认为 False。
            gate (bool): 是否使用 Gated LoRA。默认为 False。
            dora (bool): 是否使用 DoRA。默认为 False。
        '''
        super().__init__()
        self.original_layer = original_layer
        self.in_channels = original_layer.in_channels
        self.out_channels = original_layer.out_channels
        self.kernel_size = original_layer.kernel_size
        self.stride = original_layer.stride
        self.padding = original_layer.padding
        self.dilation = original_layer.dilation
        self.groups = original_layer.groups
        
        for p in self.original_layer.parameters():
            p.requires_grad = False
            
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.merged = False
        self.gate = gate
        
        if r > 0:
            self.lora_gate = nn.Parameter(torch.tensor([1.0])) if gate else None
            self.lora_a = nn.Conv2d(
                self.in_channels, r,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=False
            )
            
            self.lora_b = nn.Conv2d(
                r, self.out_channels,
                kernel_size=1, 
                stride=1, 
                padding=0,
                bias=False 
            )
            
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_a = None
            self.lora_b = None
            
        self.reset_parameters()

        self.dora = dora
        if dora and r > 0:
            # Conv2d weight: (out, in, k, k) -> norm dim=(1,2,3) for each output channel
            self.dora_m = nn.Parameter(
                self.original_layer.weight.norm(p=2, dim=(1, 2, 3), keepdim=True)
            )
        else:
            self.dora_m = None

        if merge_weights: self.merge()

    def reset_parameters(self):
        '''重置 LoRA 参数。
        
        A 卷积层使用 Kaiming Uniform 初始化，B 卷积层初始化为零。
        '''
        if self.r > 0:
            # A: Kaiming 初始化
            nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
            # B: 0 初始化
            nn.init.zeros_(self.lora_b.weight)

    def train(self, mode: bool = True):
        '''设置训练模式。

        如果进入训练模式，确保权重未合并。
        
        Args:
            mode (bool): 是否为训练模式。
        '''
        super().train(mode)
        if mode and self.merged: self.unmerge()

    def merge(self):
        '''将 LoRA 权重合并到原始卷积层权重中。
        
        使用 einsum 计算 LoRA 分支的等效卷积核并加到原始权重上。
        '''
        if self.r > 0 and not self.merged:
            weight_b = self.lora_b.weight.squeeze(3).squeeze(2) # (out, r)
            weight_a = self.lora_a.weight # (r, in, k, k)
            
            # i: out_channels, j: r, k: in_channels, m, n: kernel dims
            delta_w = torch.einsum('ij, jkmn -> ikmn', weight_b, weight_a) * self.scaling
            if self.gate: delta_w *= self.lora_gate
            
            if self.dora:
                weight = self.original_layer.weight + delta_w
                norm = weight.norm(p=2, dim=(1, 2, 3), keepdim=True)
                weight = (weight / (norm + 1e-6)) * self.dora_m
                self.original_layer.weight.data = weight
            else:
                self.original_layer.weight.data += delta_w
            
            self.merged = True

    def unmerge(self):
        '''从原始权重中减去 LoRA 权重。'''
        if self.r > 0 and self.merged:
            if self.dora:
                print("Warning: DoRA weights cannot be unmerged exactly. Original weights are lost.")
            else:
                weight_b = self.lora_b.weight.squeeze(3).squeeze(2)
                weight_a = self.lora_a.weight
                delta_w = torch.einsum('ij, jkmn -> ikmn', weight_b, weight_a) * self.scaling
                if self.gate: delta_w *= self.lora_gate
                self.original_layer.weight.data -= delta_w
            
            self.merged = False

    def forward(self, x: torch.Tensor):
        '''前向传播。
        
        Args:
            x (torch.Tensor): 输入张量。
            
        Returns:
            torch.Tensor: 输出张量。
        '''
        if self.r > 0 and self.merged:
            return self.original_layer(x)
            
        if self.dora and self.r > 0:
            weight_b = self.lora_b.weight.squeeze(3).squeeze(2) # (out, r)
            weight_a = self.lora_a.weight # (r, in, k, k)
            delta_w = torch.einsum('ij, jkmn -> ikmn', weight_b, weight_a) * self.scaling
            if self.gate: delta_w *= self.lora_gate
            
            weight = self.original_layer.weight + delta_w
            norm = weight.norm(p=2, dim=(1, 2, 3), keepdim=True)
            weight = (weight / (norm + 1e-6)) * self.dora_m
            
            return F.conv2d(
                x, weight, self.original_layer.bias, 
                self.stride, self.padding, self.dilation, self.groups
            )
            
        result = self.original_layer(x)
        
        if self.r > 0:
            x_dropped = self.lora_dropout(x)
            # Input -> Conv(in, r)[spatial] -> Conv(r, out)[1x1]
            lora_out = self.lora_b(self.lora_a(x_dropped)) * self.scaling
            if self.gate: lora_out *= self.lora_gate
            result += lora_out
            
        return result

    def __repr__(self):
        prefix = 'Gated' if self.gate else ''
        suffix = 'DoRA' if self.dora else 'LoRA'
        return f'{self.__class__.__name__}(type={prefix}{suffix}, in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups}, r={self.r}, merged={self.merged})'

class Conv1dLoRA(nn.Module):
    '''实现 Conv1d 层的 LoRA (Low-Rank Adaptation)。

    使用两个连续的卷积层模拟低秩矩阵分解：
    1. A 层: 降低通道数到 r，保持 kernel_size。
    2. B 层: 恢复通道数，使用 1x1 kernel。

    Attributes:
        original_layer (nn.Conv1d): 原始的 Conv1d 层。
        r (int): LoRA 的秩。
        lora_alpha (int): LoRA 的缩放系数。
        scaling (float): 实际缩放比例 (lora_alpha / r)。
        gate (bool): 是否使用 Gated LoRA。
        lora_gate (nn.Parameter): 门控参数。
        dora (bool): 是否使用 DoRA。
        dora_m (nn.Parameter): DoRA 的幅值向量。
        merged (bool): 权重是否已合并。
        lora_a (nn.Conv1d): 降维卷积层。
        lora_b (nn.Conv1d): 升维卷积层 (1x1)。
    '''
    def __init__(
        self, 
        original_layer: nn.Conv1d, 
        r: int = 8, 
        lora_alpha: int = 16, 
        lora_dropout: float = 0.05,
        merge_weights: bool = False,
        gate: bool = False,
        dora: bool = False
    ):
        '''初始化 Conv1dLoRA。

        Args:
            original_layer (nn.Conv1d): 原始的 Conv1d 层。
            r (int): LoRA 的秩。默认为 8。
            lora_alpha (int): LoRA 的缩放系数。默认为 16。
            lora_dropout (float): Dropout 概率。默认为 0.05。
            merge_weights (bool): 初始化时是否将 LoRA 权重合并到原始权重中。默认为 False。
            gate (bool): 是否使用 Gated LoRA。默认为 False。
            dora (bool): 是否使用 DoRA。默认为 False。
        '''
        super().__init__()
        self.original_layer = original_layer
        self.in_channels = original_layer.in_channels
        self.out_channels = original_layer.out_channels
        self.kernel_size = original_layer.kernel_size[0] # Conv1d kernel_size 是 tuple
        self.stride = original_layer.stride[0]
        self.padding = original_layer.padding[0]
        self.dilation = original_layer.dilation[0]
        self.groups = original_layer.groups
        
        # 冻结原层
        for p in self.original_layer.parameters():
            p.requires_grad = False
            
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.merged = False
        self.gate = gate
        
        if r > 0:
            self.lora_gate = nn.Parameter(torch.tensor([1.0])) if gate else None
            # A: 降维 + 空间(时序)卷积
            self.lora_a = nn.Conv1d(
                self.in_channels, r,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=False
            )
            # B: 升维 + 点卷积 (kernel=1)
            self.lora_b = nn.Conv1d(
                r, self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_a = None
            self.lora_b = None
            
        self.reset_parameters()

        self.dora = dora
        if dora and r > 0:
            # Conv1d weight: (out, in, k) -> norm dim=(1,2)
            self.dora_m = nn.Parameter(
                self.original_layer.weight.norm(p=2, dim=(1, 2), keepdim=True)
            )
        else:
            self.dora_m = None

        if merge_weights:
            self.merge()

    def reset_parameters(self):
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b.weight)

    def merge(self):
        if self.r > 0 and not self.merged:
            # B: (out, r, 1) -> (out, r)
            weight_b = self.lora_b.weight.squeeze(2)
            # A: (r, in, k)
            weight_a = self.lora_a.weight
            
            # einsum: ij(out,r), jkn(r,in,k) -> ikn(out,in,k)
            delta_w = torch.einsum('ij, jkn -> ikn', weight_b, weight_a) * self.scaling
            if self.gate: delta_w *= self.lora_gate
            
            if self.dora:
                weight = self.original_layer.weight + delta_w
                norm = weight.norm(p=2, dim=(1, 2), keepdim=True)
                weight = (weight / (norm + 1e-6)) * self.dora_m
                self.original_layer.weight.data = weight
            else:
                self.original_layer.weight.data += delta_w
            
            self.merged = True

    def unmerge(self):
        if self.r > 0 and self.merged:
            if self.dora:
                print("Warning: DoRA weights cannot be unmerged exactly. Original weights are lost.")
            else:
                weight_b = self.lora_b.weight.squeeze(2)
                weight_a = self.lora_a.weight
                delta_w = torch.einsum('ij, jkn -> ikn', weight_b, weight_a) * self.scaling
                if self.gate: delta_w *= self.lora_gate
                self.original_layer.weight.data -= delta_w
            
            self.merged = False

    def forward(self, x: torch.Tensor):
        if self.r > 0 and self.merged:
            return self.original_layer(x)
            
        if self.dora and self.r > 0:
            weight_b = self.lora_b.weight.squeeze(2)
            weight_a = self.lora_a.weight
            delta_w = torch.einsum('ij, jkn -> ikn', weight_b, weight_a) * self.scaling
            if self.gate: delta_w *= self.lora_gate
            
            weight = self.original_layer.weight + delta_w
            norm = weight.norm(p=2, dim=(1, 2), keepdim=True)
            weight = (weight / (norm + 1e-6)) * self.dora_m
            
            return F.conv1d(
                x, weight, self.original_layer.bias,
                self.stride, self.padding, self.dilation, self.groups
            )
            
        result = self.original_layer(x)
        if self.r > 0:
            x = self.lora_dropout(x)
            lora_out = self.lora_b(self.lora_a(x)) * self.scaling
            if self.gate: lora_out *= self.lora_gate
            result += lora_out
        return result

    def __repr__(self):
        prefix = 'Gated' if self.gate else ''
        suffix = 'DoRA' if self.dora else 'LoRA'
        return f'{self.__class__.__name__}(type={prefix}{suffix}, in={self.in_channels}, out={self.out_channels}, kernel={self.kernel_size}, r={self.r}, merged={self.merged})'

class EmbeddingLoRA(nn.Module):
    '''实现 Embedding 层的 LoRA (Low-Rank Adaptation)。

    通过注入低秩矩阵来适应 Embedding 权重。
    计算公式: h = W_0[idx] + (A[idx] @ B.T) * scaling

    Attributes:
        original_layer (nn.Embedding): 原始的 Embedding 层。
        r (int): LoRA 的秩。
        lora_alpha (int): LoRA 的缩放系数。
        scaling (float): 实际缩放比例 (lora_alpha / r)。
        gate (bool): 是否使用 Gated LoRA。
        lora_gate (nn.Parameter): 门控参数。
        dora (bool): 是否使用 DoRA。
        dora_m (nn.Parameter): DoRA 的幅值向量。
        merged (bool): 权重是否已合并。
        lora_a (nn.Embedding): 降维 Embedding 层 (V, r)。
        lora_b (nn.Linear): 升维 Linear 层 (r, D)。
    '''
    def __init__(
        self, 
        original_layer: nn.Embedding, 
        r: int = 8, 
        lora_alpha: int = 16, 
        merge_weights: bool = False,
        gate: bool = False,
        dora: bool = False
    ):
        '''初始化 EmbeddingLoRA。

        Args:
            original_layer (nn.Embedding): 原始的 Embedding 层。
            r (int): LoRA 的秩。默认为 8。
            lora_alpha (int): LoRA 的缩放系数。默认为 16。
            merge_weights (bool): 初始化时是否将 LoRA 权重合并到原始权重中。默认为 False。
            gate (bool): 是否使用 Gated LoRA。默认为 False。
            dora (bool): 是否使用 DoRA。默认为 False。
        '''
        super().__init__()
        self.original_layer = original_layer
        self.num_embeddings = original_layer.num_embeddings
        self.embedding_dim = original_layer.embedding_dim
        self.padding_idx = original_layer.padding_idx
        
        self.original_layer.weight.requires_grad = False
            
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.merged = False
        self.gate = gate
        
        if r > 0:
            self.lora_gate = nn.Parameter(torch.tensor([1.0])) if gate else None
            # lora_a: (num_embeddings, r)
            self.lora_a = nn.Embedding(
                self.num_embeddings, r, 
                padding_idx=self.padding_idx
            )
            # lora_b: (r, embedding_dim)
            self.lora_b = nn.Linear(r, self.embedding_dim, bias=False)
        else:
            self.lora_a = None
            self.lora_b = None
            
        self.reset_parameters()

        self.dora = dora
        if dora and r > 0:
            # Embedding weight: (V, D) -> norm dim=1
            self.dora_m = nn.Parameter(
                self.original_layer.weight.norm(p=2, dim=1, keepdim=True)
            )
        else:
            self.dora_m = None

        if merge_weights:
            self.merge()

    def reset_parameters(self):
        if self.r > 0:
            # A: 0 初始化
            nn.init.zeros_(self.lora_a.weight)
            # B: Normal 初始化
            nn.init.normal_(self.lora_b.weight, mean=0.0, std=0.02)

    def merge(self):
        if self.r > 0 and not self.merged:
            weight_b = self.lora_b.weight # (D, r)
            weight_a = self.lora_a.weight # (V, r)
            
            delta_w = (weight_a @ weight_b.T) * self.scaling
            if self.gate: delta_w *= self.lora_gate
            
            if self.dora:
                weight = self.original_layer.weight + delta_w
                norm = weight.norm(p=2, dim=1, keepdim=True)
                weight = (weight / (norm + 1e-6)) * self.dora_m
                self.original_layer.weight.data = weight
            else:
                self.original_layer.weight.data += delta_w
            
            self.merged = True

    def unmerge(self):
        if self.r > 0 and self.merged:
            if self.dora:
                print("Warning: DoRA weights cannot be unmerged exactly. Original weights are lost.")
            else:
                weight_b = self.lora_b.weight
                weight_a = self.lora_a.weight
                delta_w = (weight_a @ weight_b.T) * self.scaling
                if self.gate: delta_w *= self.lora_gate
                self.original_layer.weight.data -= delta_w
            
            self.merged = False

    def forward(self, x: torch.Tensor):
        if self.r > 0 and self.merged:
            return self.original_layer(x)
            
        if self.dora and self.r > 0:
            # DoRA embedding
            weight_b = self.lora_b.weight 
            weight_a = self.lora_a.weight 
            delta_w = (weight_a @ weight_b.T) * self.scaling
            if self.gate: delta_w *= self.lora_gate
            
            weight = self.original_layer.weight + delta_w
            norm = weight.norm(p=2, dim=1, keepdim=True)
            weight = (weight / (norm + 1e-6)) * self.dora_m
            
            return F.embedding(
                x, weight, self.padding_idx, 
                self.original_layer.max_norm, self.original_layer.norm_type,
                self.original_layer.scale_grad_by_freq, self.original_layer.sparse
            )

        result = self.original_layer(x)
        
        if self.r > 0:
            # A(x): Look up -> (Batch, Len, r)
            a_out = self.lora_a(x) 
            # B(A(x)): Linear -> (Batch, Len, Dim)
            lora_out = self.lora_b(a_out) * self.scaling
            if self.gate: lora_out *= self.lora_gate
            result += lora_out
            
        return result

    def __repr__(self):
        prefix = 'Gated' if self.gate else ''
        suffix = 'DoRA' if self.dora else 'LoRA'
        return f'{self.__class__.__name__}(type={prefix}{suffix}, num={self.num_embeddings}, dim={self.embedding_dim}, r={self.r}, merged={self.merged})'
