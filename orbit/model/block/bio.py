import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Iterable

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
        lr_state: float = 0.1,
        lr_weight: float = 1e-3,
        auto_update: bool = True,
        activation: nn.Module = nn.Tanh(),
        output_activation: nn.Module = nn.Identity()
    ):
        ''' 初始化预测编码层。

        Args:
            in_features (int): 输入特征维度。
            out_features (int): 输出特征维度（隐藏状态维度）。
            num_iter (int, optional): 推理时的迭代次数。默认为 10。
            lr_state (float, optional): 状态更新率。默认为 0.1。
            lr_weight (float, optional): 权重更新率。默认为 1e-3。
            auto_update (bool, optional): 是否在 forward 中自动更新权重。默认为 True。
            activation (nn.Module, optional): 状态激活函数。默认为 nn.Tanh()。
            output_activation (nn.Module, optional): 输出生成激活函数。默认为 nn.Identity()。
        '''
        super(PredictiveCodingLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_iter = num_iter
        self.lr_state = lr_state
        self.lr_weight = lr_weight
        self.auto_update = auto_update
        self.activation = activation
        self.output_activation = output_activation
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self._init_weights(self)
    
    def step(
        self, 
        x: torch.Tensor, 
        state: torch.Tensor, 
        mask: torch.Tensor = None, 
        top_down_input: torch.Tensor = None
    ) -> torch.Tensor:
        ''' 执行单步状态更新。
        
        使用 Autograd 自动计算能量函数相对于状态的梯度，支持非线性生成模型。
        
        Args:
            x (torch.Tensor): 输入观测值。
            state (torch.Tensor): 当前隐藏状态。
            mask (torch.Tensor, optional): 误差掩码。
            top_down_input (torch.Tensor, optional): 来自高层的预测/先验。
            
        Returns:
            torch.Tensor: 更新后的隐藏状态。
        '''
        with torch.enable_grad():
            # 启用状态的梯度追踪
            state = state.detach().requires_grad_(True)
            
            # 1. 生成预测（自顶向下生成）
            # pred_x = g(state @ W.T)
            pred_x = self.output_activation(F.linear(state, self.weight.t()))
            
            # 2. 计算能量（预测误差）
            # Energy = 0.5 * || (x - pred_x) * mask ||^2
            error = x - pred_x
            if mask is not None:
                error = error * mask
                
            energy = 0.5 * torch.sum(error ** 2)
            
            # 如果适用，添加自顶向下的先验能量
            # Energy += 0.5 * || state - top_down_input ||^2
            if top_down_input is not None:
                energy = energy + 0.5 * torch.sum((state - top_down_input) ** 2)
                
            # 3. 计算能量相对于状态的梯度
            # dEnergy/dState
            grad_state = torch.autograd.grad(energy, state)[0]
        
        # 4. 更新状态（能量梯度下降）
        # state = state - lr * grad
        new_state = state - self.lr_state * grad_state
        
        # 对状态应用激活函数
        new_state = self.activation(new_state)
        
        return new_state.detach()

    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor = None, 
        top_down_input: torch.Tensor = None
    ) -> torch.Tensor:
        ''' 前向传播。
        
        Args:
            x (torch.Tensor): 输入观测值 (Batch, ..., In_Features)。
            mask (torch.Tensor, optional): 误差掩码。
            top_down_input (torch.Tensor, optional): 来自高层的预测/先验。
            
        Returns:
            torch.Tensor: 最终的隐藏状态 (Batch, ..., Out_Features)。
        '''
        original_shape = x.shape
        if x.dim() > 2: x = x.reshape(-1, self.in_features)
        if mask is not None and mask.dim() > 2: mask = mask.reshape(-1, self.in_features)
        if top_down_input is not None and top_down_input.dim() > 2: 
            top_down_input = top_down_input.reshape(-1, self.out_features)

        # 初始化状态
        # 使用简单的前向传播进行初始化
        # 注意：这是一个近似值，理想情况下我们可能希望从随机或 0 开始
        # 但前向传播初始化可以加速收敛。
        with torch.no_grad():
            state = self.activation(F.linear(x, self.weight))
        
        # 迭代推理
        for _ in range(self.num_iter):
            state = self.step(x, state, mask, top_down_input)
            
        if self.training and self.auto_update:
            self._update_weights(x, state)
            
        if len(original_shape) > 2:
            state = state.reshape(original_shape[:-1] + (self.out_features,))
            
        return state
    
    def predict(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        ''' 执行推理并返回重构的输入（包括未观测部分）。
        
        Args:
            x (torch.Tensor): 输入观测值。
            mask (torch.Tensor, optional): 掩码。
            
        Returns:
            torch.Tensor: 重构/预测的输入 (Batch, ..., In_Features)。
        '''
        state = self.forward(x, mask)
        
        original_shape = x.shape
        if state.dim() > 2: state = state.reshape(-1, self.out_features)
        
        with torch.no_grad():
            pred_x = self.output_activation(F.linear(state, self.weight.t()))
        
        if len(original_shape) > 2:
            pred_x = pred_x.reshape(original_shape)
            
        return pred_x
    
    def _update_weights(self, x: torch.Tensor, state: torch.Tensor):
        ''' 更新权重以最小化预测误差。

        Args:
            x (torch.Tensor): 输入观测值。
            state (torch.Tensor): 隐藏状态。
        '''
        # 我们需要计算权重的梯度。
        # 因为我们正在进行手动更新，这里也可以使用 autograd。
        
        # 分离输入以确保仅进行局部学习
        x = x.detach()
        state = state.detach()
        
        # 临时启用权重的梯度（对于 Parameters 默认应该已启用）
        
        # 前向传播
        pred_x = self.output_activation(F.linear(state, self.weight.t()))
        
        # 损失
        error = x - pred_x
        loss = 0.5 * torch.sum(error ** 2)
        
        # 反向传播以获取权重的梯度
        # 我们需要先清除现有的梯度吗？
        # 因为我们在手动循环中，应该小心。
        # 但这里我们只想要这个批次的梯度。
        if self.weight.grad is not None:
            self.weight.grad.zero_()
            
        loss.backward()
        
        # 手动更新
        with torch.no_grad():
            # weight = weight - lr * grad
            # 注意：我们想要最小化预测误差。
            # 计算出的梯度是 dLoss/dWeight。
            # 所以我们减去它。
            self.weight.data -= self.lr_weight * self.weight.grad
            
            # 清除梯度
            self.weight.grad.zero_()

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
            pred_x = self.output_activation(F.linear(state, self.weight.t()))
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
        auto_update: bool = True,
        activation: nn.Module = nn.Tanh(),
        output_activations: list[nn.Module] = None
    ):
        ''' 初始化分层预测编码块。

        Args:
            in_features (int): 输入特征维度。
            hidden_dims (list[int] | int): 隐藏层维度列表。
            num_iter (int, optional): 推理迭代次数。
            lr_state (float, optional): 状态更新率。
            lr_weight (float, optional): 权重更新率。
            auto_update (bool, optional): 是否自动更新权重。
            activation (nn.Module, optional): 状态激活函数。
            output_activations (list[nn.Module], optional): 每层的输出激活函数列表。
        '''
        super(PredictiveCodingBlock, self).__init__()
        
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
            
        self.dims = [in_features] + hidden_dims
        self.num_iter = num_iter
        self.auto_update = auto_update
        
        # 默认输出激活函数
        if output_activations is None:
            output_activations = []
            # 第 0 层（输入重构）：Sigmoid（用于 [0,1] 图像）
            output_activations.append(nn.Sigmoid())
            # 后续层：Tanh（用于隐藏状态）
            for _ in range(len(self.dims) - 2):
                output_activations.append(nn.Tanh())
        
        self.layers: nn.ModuleList[PredictiveCodingLayer] = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            out_act = output_activations[i] if i < len(output_activations) else nn.Identity()
            
            self.layers.append(PredictiveCodingLayer(
                in_features=self.dims[i],
                out_features=self.dims[i+1],
                num_iter=num_iter,
                lr_state=lr_state,
                lr_weight=lr_weight,
                auto_update=False,
                activation=activation,
                output_activation=out_act
            ))
            
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        ''' 前向传播（联合推理）。
        
        Args:
            x (torch.Tensor): 输入观测值。
            mask (torch.Tensor, optional): 输入层的误差掩码。
            
        Returns:
            torch.Tensor: 第一层的隐藏状态 (用于重构输入)。
        '''
        original_shape = x.shape
        if x.dim() > 2: x = x.reshape(-1, self.dims[0])
        if mask is not None and mask.dim() > 2: mask = mask.reshape(-1, self.dims[0])
        
        # 1. 初始化状态（自底向上过程）
        states = []
        curr_input = x
        for layer in self.layers:
            # 使用前向传播初始化状态
            s = layer.activation(F.linear(curr_input, layer.weight))
            states.append(s)
            curr_input = s
            
        # 2. 联合迭代推理
        for _ in range(self.num_iter):
            # 计算所有层的自顶向下预测
            # top_down_preds[i] 是来自第 i+1 层对第 i 层的预测
            top_down_preds = [None] * len(self.layers)
            for i in range(len(self.layers) - 1):
                # 第 i+1 层预测第 i 层的状态
                # Prediction = g(State_{i+1} @ Weight_{i+1}.T)
                # 注意：第 i+1 层的权重将 State_{i+1} 映射到 State_i（其输入）
                # 我们必须使用该层的 output_activation
                with torch.no_grad():
                    top_down_preds[i] = self.layers[i+1].output_activation(
                        F.linear(states[i+1], self.layers[i+1].weight.t())
                    )
                
            # 更新所有层的状态
            new_states = []
            for i, layer in enumerate(self.layers):
                # 该层的输入
                inp = x if i == 0 else states[i-1]
                # 掩码仅适用于第一层（观测层）
                msk = mask if i == 0 else None
                
                new_s = layer.step(
                    x=inp, 
                    state=states[i], 
                    mask=msk, 
                    top_down_input=top_down_preds[i]
                )
                new_states.append(new_s)
            states = new_states
            
        # 3. 权重更新
        if self.training and self.auto_update:
            for i, layer in enumerate(self.layers):
                inp = x if i == 0 else states[i-1]
                layer._update_weights(inp, states[i])
        
        # 返回第 1 层的状态用于重构目的
        # 或者如果我们想要特征，我们可以返回最顶层的状态。
        # 但为了与 PredictiveCodingLayer 保持一致，我们返回解释输入的状态。
        state1 = states[0]
        
        if len(original_shape) > 2:
            state1 = state1.reshape(original_shape[:-1] + (self.dims[1],))
            
        return state1

    def predict(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        ''' 执行推理并返回重构的输入。 '''
        state1 = self.forward(x, mask)
        
        original_shape = x.shape
        if state1.dim() > 2: state1 = state1.reshape(-1, self.dims[1])
        
        pred_x = self.layers[0].output_activation(F.linear(state1, self.layers[0].weight.t()))
        
        if len(original_shape) > 2:
            pred_x = pred_x.reshape(original_shape)
            
        return pred_x
    
    def get_prediction_error(self, x: torch.Tensor, state1: torch.Tensor) -> torch.Tensor:
        ''' 计算输入层的预测误差。 '''
        return self.layers[0].get_prediction_error(x, state1)
