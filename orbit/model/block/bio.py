import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

from orbit.model import BaseBlock, register_model


@dataclass
class PredictiveCodingOutput:
    output: torch.Tensor
    reconstruction: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None


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
        
        # y^2 -> (B, M), 在批次上求和 -> (M)
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
        lr_state: float = 0.01,
        lr_weight: float = 1e-3,
        weight_decay: float = 0.0,
        auto_update: bool = True,
        activation: nn.Module = nn.LeakyReLU(),
        output_activation: nn.Module = nn.Identity(),
        separate_weights: bool = False
    ):
        ''' 初始化预测编码层。

        Args:
            in_features (int): 输入特征维度。
            out_features (int): 输出特征维度（隐藏状态维度）。
            num_iter (int, optional): 推理时的迭代次数。默认为 10。
            lr_state (float, optional): 状态更新率。默认为 0.01。
            lr_weight (float, optional): 权重更新率。默认为 1e-3。
            weight_decay (float, optional): 权重衰减率。默认为 0.0。
            auto_update (bool, optional): 是否在 forward 中自动更新权重。默认为 True。
            activation (nn.Module, optional): 状态激活函数。默认为 nn.LeakyReLU()。
            output_activation (nn.Module, optional): 输出生成激活函数。默认为 nn.Identity()。
            separate_weights (bool, optional): 是否使用分离的编码器和解码器权重。默认为 False。
        '''
        super(PredictiveCodingLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_iter = num_iter
        self.lr_state = lr_state
        self.lr_weight = lr_weight
        self.weight_decay = weight_decay
        self.auto_update = auto_update
        self.activation = activation
        self.output_activation = output_activation
        self.separate_weights = separate_weights
        
        if self.separate_weights:
            self.encoder = nn.Linear(in_features, out_features, bias=False)
            self.decoder = nn.Linear(out_features, in_features, bias=False)
            self.optimizer = torch.optim.Adam(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                lr=lr_weight,
                weight_decay=weight_decay
            )
        else:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.optimizer = torch.optim.Adam([self.weight], lr=lr_weight, weight_decay=weight_decay)
            
        self._init_weights(self)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        ''' 将输入投影到隐藏状态空间（线性变换）。 '''
        if self.separate_weights:
            return self.encoder(x)
        return F.linear(x, self.weight)

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        ''' 将隐藏状态投影回输入空间（线性变换）。 '''
        if self.separate_weights:
            return self.decoder(state)
        return F.linear(state, self.weight.t())

    def step(
        self, 
        x: torch.Tensor, 
        state: torch.Tensor, 
        mask: torch.Tensor = None, 
        top_down_input: torch.Tensor = None,
        feature_weights: torch.Tensor = None
    ) -> torch.Tensor:
        ''' 执行单步状态更新。
        
        使用 Autograd 自动计算能量函数相对于状态的梯度，支持非线性生成模型。
        
        Args:
            x (torch.Tensor): 输入观测值。
            state (torch.Tensor): 当前隐藏状态。
            mask (torch.Tensor, optional): 误差掩码。
            top_down_input (torch.Tensor, optional): 来自高层的预测/先验。
            feature_weights (torch.Tensor, optional): 特征权重。
            
        Returns:
            torch.Tensor: 更新后的隐藏状态。
        '''
        with torch.enable_grad():
            state = state.detach().requires_grad_(True)
            
            # pred_x = g(state @ W.T)
            pred_x = self.output_activation(self.decode(state))
            
            # Energy = 0.5 * || (x - pred_x) * mask ||^2
            error = x - pred_x
            if mask is not None:
                error = error * mask
            
            sq_error = error ** 2
            if feature_weights is not None:
                sq_error = sq_error * feature_weights
                
            energy = 0.5 * torch.sum(sq_error)
            
            # Energy += 0.5 * || state - top_down_input ||^2
            if top_down_input is not None:
                energy = energy + 0.5 * torch.sum((state - top_down_input) ** 2)
                
            # dEnergy/dState
            grad_state = torch.autograd.grad(energy, state)[0]
        
        # state = state - lr * grad
        new_state = state - self.lr_state * grad_state
        new_state = self.activation(new_state)
        
        return new_state.detach()

    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor = None, 
        top_down_input: torch.Tensor = None,
        feature_weights: torch.Tensor = None,
        num_iter: int = None
    ) -> PredictiveCodingOutput:
        ''' 前向传播。
        
        Args:
            x (torch.Tensor): 输入观测值 (Batch, ..., In_Features)。
            mask (torch.Tensor, optional): 误差掩码。
            top_down_input (torch.Tensor, optional): 来自高层的预测/先验。
            feature_weights (torch.Tensor, optional): 特征权重。
            num_iter (int, optional): 推理迭代次数。如果为 None，使用 self.num_iter。
            
        Returns:
            PredictiveCodingOutput: 包含最终隐藏状态和重构结果的对象。
        '''
        original_shape = x.shape
        if x.dim() > 2: x = x.reshape(-1, self.in_features)
        if mask is not None and mask.dim() > 2: mask = mask.reshape(-1, self.in_features)
        if top_down_input is not None and top_down_input.dim() > 2: 
            top_down_input = top_down_input.reshape(-1, self.out_features)
        if feature_weights is not None and feature_weights.dim() > 2:
            feature_weights = feature_weights.reshape(-1, self.in_features)

        with torch.no_grad():
            state = self.activation(self.encode(x))
        
        n_iter = num_iter if num_iter is not None else self.num_iter
        for _ in range(n_iter):
            state = self.step(x, state, mask, top_down_input, feature_weights)
            
        if self.training and self.auto_update:
            self._update_weights(x, state, feature_weights)
            
        if len(original_shape) > 2:
            state = state.reshape(original_shape[:-1] + (self.out_features,))
        
        with torch.no_grad():
            state_flat = state.reshape(-1, self.out_features) if state.dim() > 2 else state
            pred_x = self.output_activation(self.decode(state_flat))
            if len(original_shape) > 2:
                pred_x = pred_x.reshape(original_shape)

        return PredictiveCodingOutput(
            output=state,
            reconstruction=pred_x
        )
    
    def predict(self, x: torch.Tensor, mask: torch.Tensor = None, feature_weights: torch.Tensor = None, num_iter: int = None) -> torch.Tensor:
        ''' 执行推理并返回重构的输入（包括未观测部分）。
        
        Args:
            x (torch.Tensor): 输入观测值。
            mask (torch.Tensor, optional): 掩码。
            feature_weights (torch.Tensor, optional): 特征权重。
            num_iter (int, optional): 推理迭代次数。
            
        Returns:
            torch.Tensor: 重构/预测的输入 (Batch, ..., In_Features)。
        '''
        pc_output = self.forward(x, mask, feature_weights=feature_weights, num_iter=num_iter)
        return pc_output.reconstruction
    
    def _update_weights(self, x: torch.Tensor, state: torch.Tensor, feature_weights: torch.Tensor = None):
        ''' 更新权重以最小化预测误差。

        Args:
            x (torch.Tensor): 输入观测值。
            state (torch.Tensor): 隐藏状态。
            feature_weights (torch.Tensor, optional): 特征权重。
        '''
        x = x.detach()
        state = state.detach()
        
        self.optimizer.zero_grad()
        
        if self.separate_weights:
            # || x - decoder(state) ||^2
            pred_x = self.output_activation(self.decoder(state))
            error = x - pred_x
            
            sq_error = error ** 2
            if feature_weights is not None:
                sq_error = sq_error * feature_weights
            loss_decoder = 0.5 * torch.sum(sq_error)
            
            # || state - encoder(x) ||^2 (Amortized Inference)
            pred_state = self.activation(self.encoder(x))
            loss_encoder = 0.5 * torch.sum((state - pred_state) ** 2)
            
            loss = loss_decoder + loss_encoder
            loss.backward()
        else:
            pred_x = self.output_activation(F.linear(state, self.weight.t()))
            
            error = x - pred_x
            sq_error = error ** 2
            if feature_weights is not None:
                sq_error = sq_error * feature_weights
            loss = 0.5 * torch.sum(sq_error)
            
            loss.backward()
            
        self.optimizer.step()

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
            pred_x = self.output_activation(self.decode(state))
            return x - pred_x


@register_model()
class PredictiveCodingBlock(BaseBlock):
    ''' 分层预测编码块。
    
    自动管理多层 PredictiveCodingLayer，实现分层预测编码网络。
    支持任意深度的层级结构和联合推理。
    '''
    def __init__(
        self,
        in_features: int,
        hidden_dims: list[int] | int,
        num_iter: int = 10,
        lr_state: float = 0.1,
        lr_weight: float = 1e-3,
        weight_decay: float = 0.0,
        auto_update: bool = True,
        activation: nn.Module = nn.Tanh(),
        output_activations: list[nn.Module] = None,
        separate_weights: bool = False
    ):
        ''' 初始化分层预测编码块。

        Args:
            in_features (int): 输入特征维度。
            hidden_dims (list[int] | int): 隐藏层维度列表。
            num_iter (int, optional): 推理迭代次数。
            lr_state (float, optional): 状态更新率。
            lr_weight (float, optional): 权重更新率。
            weight_decay (float, optional): 权重衰减率。
            auto_update (bool, optional): 是否自动更新权重。
            activation (nn.Module, optional): 状态激活函数。
            output_activations (list[nn.Module], optional): 每层的输出激活函数列表。
            separate_weights (bool, optional): 是否使用分离的编码器和解码器权重。
        '''
        super(PredictiveCodingBlock, self).__init__()
        
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
            
        self.dims = [in_features] + hidden_dims
        self.num_iter = num_iter
        self.auto_update = auto_update
        
        if output_activations is None:
            output_activations = []
            output_activations.append(nn.LeakyReLU())
            for _ in range(len(self.dims) - 2):
                output_activations.append(nn.LeakyReLU())
        
        self.layers: nn.ModuleList[PredictiveCodingLayer] = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            out_act = output_activations[i] if i < len(output_activations) else nn.Identity()
            
            self.layers.append(PredictiveCodingLayer(
                in_features=self.dims[i],
                out_features=self.dims[i+1],
                num_iter=num_iter,
                lr_state=lr_state,
                lr_weight=lr_weight,
                weight_decay=weight_decay,
                auto_update=False,
                activation=activation,
                output_activation=out_act,
                separate_weights=separate_weights
            ))
            
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, feature_weights: torch.Tensor = None, num_iter: int = None) -> PredictiveCodingOutput:
        ''' 前向传播（联合推理）。
        
        Args:
            x (torch.Tensor): 输入观测值。
            mask (torch.Tensor, optional): 输入层的误差掩码。
            feature_weights (torch.Tensor, optional): 输入层的特征权重。
            num_iter (int, optional): 推理迭代次数。如果为 None，使用 self.num_iter。
            
        Returns:
            PredictiveCodingOutput: 包含第一层隐藏状态、重构结果和所有层状态的对象。
        '''
        original_shape = x.shape
        if x.dim() > 2: x = x.reshape(-1, self.dims[0])
        if mask is not None and mask.dim() > 2: mask = mask.reshape(-1, self.dims[0])
        if feature_weights is not None and feature_weights.dim() > 2:
            feature_weights = feature_weights.reshape(-1, self.dims[0])
        
        states = []
        curr_input = x
        for layer in self.layers:
            s = layer.activation(layer.encode(curr_input))
            states.append(s)
            curr_input = s
            
        n_iter = num_iter if num_iter is not None else self.num_iter
        for _ in range(n_iter):
            top_down_preds = [None] * len(self.layers)
            for i in range(len(self.layers) - 1):
                with torch.no_grad():
                    top_down_preds[i] = self.layers[i+1].output_activation(
                        self.layers[i+1].decode(states[i+1])
                    )
                
            new_states = []
            for i, layer in enumerate(self.layers):
                inp = x if i == 0 else states[i-1]
                
                msk = mask if i == 0 else None
                fw = feature_weights if i == 0 else None
                
                new_s = layer.step(
                    x=inp, 
                    state=states[i], 
                    mask=msk, 
                    top_down_input=top_down_preds[i],
                    feature_weights=fw
                )
                new_states.append(new_s)
            states = new_states
            
        if self.training and self.auto_update:
            for i, layer in enumerate(self.layers):
                inp = x if i == 0 else states[i-1]
                fw = feature_weights if i == 0 else None
                layer._update_weights(inp, states[i], feature_weights=fw)
        
        state1 = states[0]
        
        if len(original_shape) > 2:
            state1 = state1.reshape(original_shape[:-1] + (self.dims[1],))
        
        with torch.no_grad():
            state1_flat = state1.reshape(-1, self.dims[1]) if state1.dim() > 2 else state1
            pred_x = self.layers[0].output_activation(self.layers[0].decode(state1_flat))
            if len(original_shape) > 2:
                pred_x = pred_x.reshape(original_shape)
            
        return PredictiveCodingOutput(
            output=state1,
            reconstruction=pred_x,
            hidden_states=tuple(states)
        )

    def predict(self, x: torch.Tensor, mask: torch.Tensor = None, feature_weights: torch.Tensor = None, num_iter: int = None) -> torch.Tensor:
        ''' 执行推理并返回重构的输入。 '''
        pc_output = self.forward(x, mask, feature_weights=feature_weights, num_iter=num_iter)
        return pc_output.reconstruction
    
    def get_prediction_error(self, x: torch.Tensor, state1: torch.Tensor) -> torch.Tensor:
        ''' 计算输入层的预测误差。 '''
        return self.layers[0].get_prediction_error(x, state1)
