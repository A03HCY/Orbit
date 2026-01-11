import torch
import torch.nn as nn
import torch.nn.functional as F

from orbit.model import BaseBlock, register_model


@register_model()
class HebianLayer(BaseBlock):
    ''' Hebbian Learning Layer.

    实现基于 Hebbian 规则的无监督学习层。支持标准 Hebbian 规则和 Oja 规则。
    '''
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        lr: float = 1e-3,
        mode: str = 'oja',
        bias: bool = True,
        auto_update: bool = True
    ):
        ''' 初始化 Hebbian 学习层。

        Args:
            in_features (int): 输入特征维度。
            out_features (int): 输出特征维度。
            lr (float, optional): Hebbian 学习率。默认为 1e-3。
            mode (str, optional): 更新模式，可选 'basic' 或 'oja'。默认为 'oja'。
            bias (bool, optional): 是否使用偏置。默认为 True。
            auto_update (bool, optional): 是否在 forward 中自动更新权重。默认为 True。
        '''
        super(HebianLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lr = lr
        self.mode = mode.lower()
        self.auto_update = auto_update
        
        if self.mode not in ['basic', 'oja']:
            raise ValueError(f"Unsupported mode: {mode}. Must be 'basic' or 'oja'.")

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self._init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' 前向传播。

        Args:
            x (torch.Tensor): 输入张量 (Batch, ..., In_Features)。

        Returns:
            torch.Tensor: 输出张量 (Batch, ..., Out_Features)。
        '''
        y = F.linear(x, self.weight, self.bias)
        
        if self.training and self.auto_update:
            if x.dim() > 2:
                x_flat = x.reshape(-1, x.size(-1))
                y_flat = y.reshape(-1, y.size(-1))
                self._update_weights(x_flat, y_flat)
            else:
                self._update_weights(x, y)
                
        return y

    @torch.no_grad()
    def _update_weights(self, x: torch.Tensor, y: torch.Tensor):
        ''' 执行权重更新。 '''
        if self.mode == 'basic':
            self._basic_update(x, y)
        elif self.mode == 'oja':
            self._oja_update(x, y)

    @torch.no_grad()
    def _basic_update(self, x: torch.Tensor, y: torch.Tensor):
        ''' 执行标准 Hebbian 更新规则。

        Args:
            x (torch.Tensor): 输入张量。
            y (torch.Tensor): 输出张量。
        '''
        batch_size = x.size(0)
        
        # y^T * x -> (M, N)
        grad_w = torch.matmul(y.t(), x)
        
        self.weight.data += self.lr * grad_w / batch_size
        
        if self.bias is not None:
            # db = lr * sum(y)
            grad_b = y.sum(dim=0)
            self.bias.data += self.lr * grad_b / batch_size

    @torch.no_grad()
    def _oja_update(self, x: torch.Tensor, y: torch.Tensor):
        ''' 执行 Oja 更新规则。

        Oja 规则通过归一化防止权重无限增长。

        Args:
            x (torch.Tensor): 输入张量。
            y (torch.Tensor): 输出张量。
        '''
        batch_size = x.size(0)
        
        # y^T * x -> (M, N)
        yx = torch.matmul(y.t(), x)
        
        # y^2 -> (B, M), sum over batch -> (M)
        y_sq = torch.sum(y ** 2, dim=0)
        
        # (M, 1) * (M, N) -> (M, N)
        grad_w = yx - y_sq.unsqueeze(1) * self.weight
        
        self.weight.data += self.lr * grad_w / batch_size
        
        if self.bias is not None:
            grad_b = y.sum(dim=0) - y_sq * self.bias
            self.bias.data += self.lr * grad_b / batch_size


@register_model()
class PredictiveCodingLayer(BaseBlock):
    ''' Predictive Coding Layer.
    
    实现基于预测编码原理的层。该层维护一个内部状态（表示），
    并通过最小化预测误差来更新状态。
    '''
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_iter: int = 10,
        lr_state: float = 0.1,
        lr_weight: float = 1e-3,
        auto_update: bool = True,
        activation: nn.Module = nn.Tanh()
    ):
        ''' 初始化预测编码层。

        Args:
            in_features (int): 输入特征维度。
            out_features (int): 输出特征维度（隐藏状态维度）。
            num_iter (int, optional): 推理时的迭代次数。默认为 10。
            lr_state (float, optional): 状态更新率。默认为 0.1。
            lr_weight (float, optional): 权重更新率。默认为 1e-3。
            auto_update (bool, optional): 是否在 forward 中自动更新权重。默认为 True。
            activation (nn.Module, optional): 激活函数。默认为 nn.Tanh()。
        '''
        super(PredictiveCodingLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_iter = num_iter
        self.lr_state = lr_state
        self.lr_weight = lr_weight
        self.auto_update = auto_update
        self.activation = activation
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self._init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' 前向传播。
        
        Args:
            x (torch.Tensor): 输入观测值 (Batch, ..., In_Features)。
            
        Returns:
            torch.Tensor: 最终的隐藏状态 (Batch, ..., Out_Features)。
        '''
        original_shape = x.shape
        if x.dim() > 2: x = x.reshape(-1, self.in_features)
        state = self.activation(F.linear(x, self.weight))
        
        for _ in range(self.num_iter):
            pred_x = F.linear(state, self.weight.t())
            error = x - pred_x
            delta_state = torch.matmul(error, self.weight.t())
            
            state = state + self.lr_state * delta_state
            state = self.activation(state)
            
        if self.training and self.auto_update:
            self._update_weights(x, state)
            
        if len(original_shape) > 2:
            state = state.reshape(original_shape[:-1] + (self.out_features,))
            
        return state
    
    @torch.no_grad()
    def _update_weights(self, x: torch.Tensor, state: torch.Tensor):
        ''' 更新权重以最小化预测误差。

        Args:
            x (torch.Tensor): 输入观测值。
            state (torch.Tensor): 隐藏状态。
        '''
        batch_size = x.size(0)
        
        pred_x = F.linear(state, self.weight.t())
        error = x - pred_x
        
        grad = torch.matmul(state.t(), error)
        
        self.weight.data += self.lr_weight * grad / batch_size

    def get_prediction_error(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        ''' 计算预测误差。

        Args:
            x (torch.Tensor): 输入观测值。
            state (torch.Tensor): 隐藏状态。

        Returns:
            torch.Tensor: 预测误差 (x - pred_x)。
        '''
        if x.dim() > 2:
            x = x.reshape(-1, self.in_features)
            state = state.reshape(-1, self.out_features)
            
        with torch.no_grad():
            pred_x = F.linear(state, self.weight.t())
            return x - pred_x
