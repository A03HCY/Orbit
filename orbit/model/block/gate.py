import torch
import torch.nn as nn
import torch.nn.functional as F

from orbit.model import BaseBlock, register_model
from orbit.model.block.mlp import MLP


class BaseGate(BaseBlock):
    ''' 门控模块基类。

    提供通用的 MLP/Linear 变换逻辑。
    '''

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        use_mlp: bool = False,
        hidden_features: int = None
    ):
        ''' 初始化 BaseGate。

        Args:
            in_features (int): 输入特征维度。
            out_features (int): 输出特征维度。
            bias (bool, optional): 是否使用偏置。默认为 True。
            use_mlp (bool, optional): 是否使用 MLP 进行变换。默认为 False。
            hidden_features (int, optional): MLP 的隐藏层维度。仅在 use_mlp=True 时有效。默认为 None。
        '''
        super(BaseGate, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.use_mlp = use_mlp
        self.hidden_features = hidden_features

        if use_mlp:
            hidden_features = hidden_features or in_features
            self.mlp = MLP(
                in_features=in_features,
                hidden_features=hidden_features,
                out_features=out_features,
                use_gate=False,
                dropout=0.0
            )
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_mlp:
            return self.mlp(x)
        else:
            return self.linear(x)

    def _get_repr_args(self) -> list[str]:
        args = [
            f"in_features={self.in_features}",
            f"out_features={self.out_features}",
            f"bias={self.bias}",
            f"use_mlp={self.use_mlp}"
        ]
        if self.use_mlp and self.hidden_features is not None:
            args.append(f"hidden_features={self.hidden_features}")
        return args

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(self._get_repr_args())})"


@register_model()
class SigmoidGate(BaseGate):
    ''' Sigmoid 门控模块。

    实现: Sigmoid(Linear(x)) 或 Sigmoid(MLP(x))
    用于生成 0 到 1 之间的门控值。
    '''

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' 前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 门控值，范围 [0, 1]。
        '''
        return torch.sigmoid(self._transform(x))


@register_model()
class TanhGate(BaseGate):
    ''' Tanh 门控模块。

    实现: Tanh(Linear(x)) 或 Tanh(MLP(x))
    用于生成 -1 到 1 之间的门控值。
    '''

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' 前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 门控值，范围 [-1, 1]。
        '''
        return torch.tanh(self._transform(x))


@register_model()
class SoftmaxGate(BaseGate):
    ''' Softmax 门控模块。

    实现: Softmax(Linear(x)) 或 Softmax(MLP(x))
    用于生成和为 1 的门控值。
    '''

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        dim: int = -1,
        temperature: float = 1.0,
        bias: bool = True,
        use_mlp: bool = False,
        hidden_features: int = None
    ):
        ''' 初始化 SoftmaxGate。

        Args:
            in_features (int): 输入特征维度。
            out_features (int): 输出特征维度。
            dim (int, optional): Softmax 操作的维度。默认为 -1。
            temperature (float, optional): 温度系数，用于控制分布的平滑程度。默认为 1.0。
            bias (bool, optional): 是否使用偏置。默认为 True。
            use_mlp (bool, optional): 是否使用 MLP 进行变换。默认为 False。
            hidden_features (int, optional): MLP 的隐藏层维度。仅在 use_mlp=True 时有效。默认为 None。
        '''
        super(SoftmaxGate, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            use_mlp=use_mlp,
            hidden_features=hidden_features
        )
        self.dim = dim
        self.temperature = temperature

    def _get_repr_args(self) -> list[str]:
        args = super()._get_repr_args()
        args.append(f"dim={self.dim}")
        args.append(f"temperature={self.temperature}")
        return args

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' 前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 门控值，和为 1。
        '''
        x = self._transform(x)
        return F.softmax(x / self.temperature, dim=self.dim)


@register_model()
class GLUGate(BaseBlock):
    ''' Gated Linear Unit (GLU) 门控模块。

    支持多种激活函数 (Sigmoid, Tanh, ReLU, GELU, SiLU)。
    实现: (x * W + b) * Activation(x * V + c)
    '''

    def __init__(
        self, 
        in_features: int, 
        hidden_features: int,
        out_features: int,
        activation: str = 'sigmoid',
        bias: bool = True
    ):
        ''' 初始化 GLUGate。

        Args:
            in_features (int): 输入维度。
            hidden_features (int): 隐藏层维度 (投影维度)。
            out_features (int): 输出维度。
            activation (str, optional): 激活函数类型 ('sigmoid', 'tanh', 'relu', 'gelu', 'silu')。默认为 'sigmoid'。
            bias (bool, optional): Linear 层是否使用偏置。默认为 True。
        '''
        super(GLUGate, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.activation_name = activation
        self.bias = bias

        self.proj = nn.Linear(in_features, hidden_features * 2, bias=bias)
        self.out_proj = nn.Linear(hidden_features, out_features, bias=bias)

        if activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def __repr__(self):
        args = [
            f"in_features={self.in_features}",
            f"hidden_features={self.hidden_features}",
            f"out_features={self.out_features}",
            f"activation='{self.activation_name}'",
            f"bias={self.bias}"
        ]
        return f"{self.__class__.__name__}({', '.join(args)})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' 前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        '''
        x = self.proj(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * self.act(gate)
        return self.out_proj(x)


@register_model()
class TopKGate(BaseGate):
    ''' Top-K 门控模块。

    只保留 Top-K 个最大的值，其余置零。
    支持返回详细路由信息以供 MoE 使用。
    '''

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        k: int = 1,
        bias: bool = True,
        use_mlp: bool = False,
        hidden_features: int = None,
        post_softmax: bool = False
    ):
        ''' 初始化 TopKGate。

        Args:
            in_features (int): 输入特征维度。
            out_features (int): 输出特征维度。
            k (int, optional): 保留的 Top-K 值的数量。默认为 1。
            bias (bool, optional): 是否使用偏置。默认为 True。
            use_mlp (bool, optional): 是否使用 MLP 进行变换。默认为 False。
            hidden_features (int, optional): MLP 的隐藏层维度。仅在 use_mlp=True 时有效。默认为 None。
            post_softmax (bool, optional): 是否在 Top-K 选择后对值进行 Softmax 归一化。默认为 False。
        '''
        super(TopKGate, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            use_mlp=use_mlp,
            hidden_features=hidden_features
        )
        self.k = k
        self.post_softmax = post_softmax

    def _get_repr_args(self) -> list[str]:
        args = super()._get_repr_args()
        args.append(f"k={self.k}")
        args.append(f"post_softmax={self.post_softmax}")
        return args

    def forward(self, x: torch.Tensor, return_meta: bool = False) -> torch.Tensor:
        ''' 前向传播。

        Args:
            x (torch.Tensor): 输入张量。
            return_meta (bool, optional): 是否返回元数据 (logits, indices, values)。默认为 False。

        Returns:
            torch.Tensor: 门控后的输出张量。如果 return_meta=True，则返回 (logits, indices, values)。
        '''
        logits = self._transform(x)
        
        topk_values, topk_indices = torch.topk(logits, self.k, dim=-1)
        
        if self.post_softmax:
            topk_values = F.softmax(topk_values, dim=-1)
        
        if return_meta:
            return logits, topk_indices, topk_values

        output = torch.zeros_like(logits)
        output.scatter_(-1, topk_indices, topk_values)
        
        return output


@register_model()
class ContextGate(BaseBlock):
    ''' 上下文门控模块 (Context Gating)。

    实现: x * Sigmoid(MLP(x))
    使用输入自身的上下文信息来调节特征。
    '''

    def __init__(
        self, 
        in_features: int, 
        hidden_features: int = None,
        bias: bool = True
    ):
        ''' 初始化 ContextGate。

        Args:
            in_features (int): 输入维度。
            hidden_features (int, optional): MLP 的隐藏层维度。默认为 in_features / 2。
            bias (bool, optional): Linear 层是否使用偏置。默认为 True。
        '''
        super(ContextGate, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features or (in_features // 2)
        self.bias = bias

        self.mlp = nn.Sequential(
            nn.Linear(in_features, self.hidden_features, bias=bias),
            nn.ReLU(),
            nn.Linear(self.hidden_features, in_features, bias=bias),
            nn.Sigmoid()
        )

    def __repr__(self):
        args = [
            f"in_features={self.in_features}",
            f"hidden_features={self.hidden_features}",
            f"bias={self.bias}"
        ]
        return f"{self.__class__.__name__}({', '.join(args)})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' 前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 门控后的输出张量。
        '''
        return x * self.mlp(x)
