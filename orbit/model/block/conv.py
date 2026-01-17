import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Union

from orbit.model import BaseBlock, register_model


def calculate_causal_layer(step: int, kernel_size: int = 3) -> Tuple[int, int]:
    '''
    计算因果卷积所需的层数和感受野。

    Args:
        step (int): 目标序列长度或时间步数。
        kernel_size (int, optional): 卷积核大小。默认为 3。

    Returns:
        tuple[int, int]:
            - L (int): 所需的层数。
            - R (int): 最终的感受野大小。

    Raises:
        ValueError: 如果 kernel_size <= 1。
    '''
    if kernel_size <= 1:
        raise ValueError('kernel_size must be greater than 1')
    L = math.ceil(math.log2((step - 1) / (kernel_size - 1) + 1))
    R = 1 + (kernel_size - 1) * (2 ** L - 1)
    return int(L), R


@register_model()
class CausalConv1d(BaseBlock):
    '''
    因果一维卷积层 (Causal 1D Convolution)。
    
    通过膨胀卷积 (Dilated Convolution) 和因果填充 (Causal Padding) 实现，
    确保当前时刻的输出仅依赖于当前及过去的输入，不看未来。
    常用于时序数据处理和波形生成 (如 WaveNet)。
    '''

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3, 
        dilation: int = 1, 
        norm: str = None,
        activation: str = 'leaky_relu',
        leaky_relu: float = 0.1, 
        use_res: bool = True, 
        dropout: float = 0.2
    ):
        '''
        初始化 CausalConv1d 模块。

        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            kernel_size (int, optional): 卷积核大小。默认为 3。
            dilation (int, optional): 膨胀系数。默认为 1。
            norm (str, optional): 归一化类型。默认为 None。
            activation (str, optional): 激活函数类型。默认为 'leaky_relu'。
            leaky_relu (float, optional): LeakyReLU 的负斜率 (仅当 activation='leaky_relu' 时有效)。默认为 0.1。
            use_res (bool, optional): 是否使用残差连接。默认为 True。
            dropout (float, optional): Dropout 概率。默认为 0.2。
        '''
        super(CausalConv1d, self).__init__()
        
        self.padding = (kernel_size - 1) * dilation
        self.use_res = use_res
        
        self.block = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            dim=1,
            norm=norm,
            activation=activation,
            dropout=dropout
        )
        
        if activation == 'leaky_relu' and isinstance(self.block.act, nn.LeakyReLU) and leaky_relu != 0.1:
            self.block.act = nn.LeakyReLU(leaky_relu, inplace=True)

        self.block.conv = nn.utils.parametrizations.weight_norm(self.block.conv)
        
        self.downsample = None
        if use_res and in_channels != out_channels:
            self.downsample = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                dim=1,
                norm=None,
                activation=None,
                dropout=0.0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        前向传播。

        Args:
            x (torch.Tensor): 输入张量。Shape: [Batch, in_channels, Seq_Len]

        Returns:
            torch.Tensor: 输出张量。Shape: [Batch, out_channels, Seq_Len]
        '''
        residual = x
        x = F.pad(x, (self.padding, 0))
        x = self.block(x)
        
        if self.use_res:
            if self.downsample is not None:
                residual = self.downsample(residual)
            x = x + residual
            
        return x
    
    @staticmethod
    def auto_block(
        in_channels: int, 
        out_channels: int, 
        step: int, 
        kernel_size: int = 3, 
        norm: str = None,
        activation: str = 'leaky_relu',
        leaky_relu: float = 0.1, 
        use_res: bool = True, 
        dropout: float = 0.2
    ) -> nn.Sequential:
        '''
        自动构建多层因果卷积块以覆盖指定的时间步长。

        根据目标步长 step 自动计算所需的层数和膨胀系数，构建一个 nn.Sequential 模型。
        膨胀系数随层数指数增长 (1, 2, 4, 8, ...)。

        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            step (int): 目标覆盖的时间步长 (感受野)。
            kernel_size (int, optional): 卷积核大小。默认为 3。
            norm (str, optional): 归一化类型。默认为 None。
            activation (str, optional): 激活函数类型。默认为 'leaky_relu'。
            leaky_relu (float, optional): LeakyReLU 的负斜率。默认为 0.1。
            use_res (bool, optional): 是否使用残差连接。默认为 True。
            dropout (float, optional): Dropout 概率。默认为 0.2。

        Returns:
            nn.Sequential: 包含多个 CausalConv1d 层的序列模型。
        '''
        layers, _ = calculate_causal_layer(step, kernel_size)
        model = []
        for i in range(layers):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else out_channels

            model.append(CausalConv1d(
                in_channels=in_ch, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                dilation=dilation, 
                norm=norm,
                activation=activation,
                leaky_relu=leaky_relu, 
                use_res=use_res, 
                dropout=dropout
            ))

        return nn.Sequential(*model)


@register_model()
class ConvBlock(BaseBlock):
    '''
    通用卷积块 (Conv-Norm-Act-Dropout)。

    支持 1D、2D 和 3D 卷积，以及多种归一化和激活函数配置。
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...], str] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1,
        bias: bool = True,
        dim: int = 2,
        norm: str = 'batch',
        activation: str = 'relu',
        dropout: float = 0.0,
        pre_norm: bool = False,
    ):
        '''
        初始化 ConvBlock。

        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            kernel_size (Union[int, Tuple[int, ...]], optional): 卷积核大小。默认为 3。
            stride (Union[int, Tuple[int, ...]], optional): 步幅。默认为 1。
            padding (Union[int, Tuple[int, ...], str], optional): 填充。默认为 0。
            dilation (Union[int, Tuple[int, ...]], optional): 膨胀系数。默认为 1。
            groups (int, optional): 分组卷积组数。默认为 1。
            bias (bool, optional): 是否使用偏置。默认为 True。
            dim (int, optional): 卷积维度 (1, 2, 3)。默认为 2。
            norm (str, optional): 归一化类型 ('batch', 'group', 'layer', 'instance', None)。默认为 'batch'。
            activation (str, optional): 激活函数类型 ('relu', 'leaky_relu', 'gelu', 'silu', 'tanh', 'sigmoid', None)。默认为 'relu'。
            dropout (float, optional): Dropout 概率。默认为 0.0。
            pre_norm (bool, optional): 是否使用 Pre-Norm (Norm-Conv-Act) 结构。默认为 False (Conv-Norm-Act)。
        '''
        super(ConvBlock, self).__init__()
        
        self.pre_norm = pre_norm
        
        if dim == 1:
            conv_class = nn.Conv1d
        elif dim == 2:
            conv_class = nn.Conv2d
        elif dim == 3:
            conv_class = nn.Conv3d
        else:
            raise ValueError(f'Unsupported dimension: {dim}')
            
        self.conv = conv_class(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias
        )
        
        self.norm = None
        if norm is not None:
            norm_channels = in_channels if pre_norm else out_channels
            if norm == 'batch':
                if dim == 1: self.norm = nn.BatchNorm1d(norm_channels)
                elif dim == 2: self.norm = nn.BatchNorm2d(norm_channels)
                elif dim == 3: self.norm = nn.BatchNorm3d(norm_channels)
            elif norm == 'group':
                num_groups = 32 if norm_channels % 32 == 0 else min(norm_channels, 8)
                self.norm = nn.GroupNorm(num_groups, norm_channels)
            elif norm == 'layer':
                self.norm = nn.GroupNorm(1, norm_channels) 
            elif norm == 'instance':
                if dim == 1: self.norm = nn.InstanceNorm1d(norm_channels)
                elif dim == 2: self.norm = nn.InstanceNorm2d(norm_channels)
                elif dim == 3: self.norm = nn.InstanceNorm3d(norm_channels)
            else:
                raise ValueError(f'Unsupported normalization: {norm}')

        self.act = None
        if activation is not None:
            if activation == 'relu':
                self.act = nn.ReLU(inplace=True)
            elif activation == 'leaky_relu':
                self.act = nn.LeakyReLU(0.1, inplace=True)
            elif activation == 'gelu':
                self.act = nn.GELU()
            elif activation == 'silu':
                self.act = nn.SiLU(inplace=True)
            elif activation == 'tanh':
                self.act = nn.Tanh()
            elif activation == 'sigmoid':
                self.act = nn.Sigmoid()
            else:
                raise ValueError(f'Unsupported activation: {activation}')

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        '''
        if self.pre_norm:
            if self.norm: x = self.norm(x)
            if self.act: x = self.act(x)
            x = self.conv(x)
            if self.dropout: x = self.dropout(x)
        else:
            x = self.conv(x)
            if self.norm: x = self.norm(x)
            if self.act: x = self.act(x)
            if self.dropout: x = self.dropout(x)
            
        return x


@register_model()
class DepthwiseSeparableConv(BaseBlock):
    '''
    深度可分离卷积块 (Depthwise Separable Convolution)。
    
    由一个 Depthwise Conv（逐通道卷积）和一个 Pointwise Conv（逐点卷积）组成。
    在保持特征提取能力的同时，大幅降低参数量和计算开销。
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...], str] = 1,
        dilation: Union[int, Tuple[int, ...]] = 1,
        bias: bool = False,
        dim: int = 2,
        norm: str = 'batch',
        activation: str = 'relu',
        dropout: float = 0.0,
        use_res: bool = True,
    ):
        '''
        初始化 DepthwiseSeparableConv。

        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            kernel_size: 卷积核大小。
            stride: 步幅。
            padding: 填充。
            dilation: 膨胀系数。
            bias (bool): 是否使用偏置。
            dim (int): 卷积维度 (1, 2, 3)。
            norm (str): 归一化类型。
            activation (str): 激活函数类型。
            dropout (float): Dropout 概率。
            use_res (bool): 是否在输入输出通道一致且步幅为1时使用残差连接。
        '''
        super(DepthwiseSeparableConv, self).__init__()

        self.depthwise = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            dim=dim,
            norm=norm,
            activation=activation,
            dropout=0.0
        )

        self.pointwise = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            dim=dim,
            norm=norm,
            activation=activation,
            dropout=dropout
        )

        self.use_res = use_res and (in_channels == out_channels) and (stride == 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        前向传播。
        '''
        identity = x
        
        out = self.depthwise(x)
        out = self.pointwise(out)

        if self.use_res:
            out += identity
            
        return out


@register_model()
class ResBasicBlock(BaseBlock):
    '''
    残差基本块 (Residual Basic Block)。

    由两个卷积层组成，支持标准 ResNet 和 Pre-activation ResNet 变体。
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...], str] = 1,
        dilation: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1,
        bias: bool = False,
        dim: int = 2,
        norm: str = 'batch',
        activation: str = 'relu',
        dropout: float = 0.0,
        variant: str = 'original',
    ):
        '''
        初始化 ResBasicBlock。

        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            kernel_size (Union[int, Tuple[int, ...]], optional): 卷积核大小。默认为 3。
            stride (Union[int, Tuple[int, ...]], optional): 步幅。默认为 1。
            padding (Union[int, Tuple[int, ...], str], optional): 填充。默认为 1。
            dilation (Union[int, Tuple[int, ...]], optional): 膨胀系数。默认为 1。
            groups (int, optional): 分组卷积组数。默认为 1。
            bias (bool, optional): 是否使用偏置。默认为 False。
            dim (int, optional): 卷积维度 (1, 2, 3)。默认为 2。
            norm (str, optional): 归一化类型。默认为 'batch'。
            activation (str, optional): 激活函数类型。默认为 'relu'。
            dropout (float, optional): Dropout 概率。默认为 0.0。
            variant (str, optional): 变体类型 ('original', 'pre_act')。默认为 'original'。
        '''
        super(ResBasicBlock, self).__init__()
        
        self.variant = variant
        self.activation = activation
        
        self.act = None
        if variant == 'original' and activation is not None:
            if activation == 'relu':
                self.act = nn.ReLU(inplace=True)
            elif activation == 'leaky_relu':
                self.act = nn.LeakyReLU(0.1, inplace=True)
            elif activation == 'gelu':
                self.act = nn.GELU()
            elif activation == 'silu':
                self.act = nn.SiLU(inplace=True)
            elif activation == 'tanh':
                self.act = nn.Tanh()
            elif activation == 'sigmoid':
                self.act = nn.Sigmoid()

        if variant == 'original':
            # Conv1: Conv-Norm-Act
            self.conv1 = ConvBlock(
                in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, dim, norm, activation, dropout, pre_norm=False
            )
            # Conv2: Conv-Norm
            self.conv2 = ConvBlock(
                out_channels, out_channels, kernel_size, 1, padding, dilation, groups, bias, dim, norm, activation=None, dropout=dropout, pre_norm=False
            )
        elif variant == 'pre_act':
            # Conv1: Norm-Act-Conv
            self.conv1 = ConvBlock(
                in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, dim, norm, activation, dropout, pre_norm=True
            )
            # Conv2: Norm-Act-Conv
            self.conv2 = ConvBlock(
                out_channels, out_channels, kernel_size, 1, padding, dilation, groups, bias, dim, norm, activation, dropout, pre_norm=True
            )
        else:
            raise ValueError(f'Unsupported variant: {variant}')

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            if variant == 'original':
                self.downsample = ConvBlock(
                    in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dim=dim, norm=norm, activation=None, bias=bias, pre_norm=False
                )
            elif variant == 'pre_act':
                self.downsample = ConvBlock(
                    in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dim=dim, norm=None, activation=None, bias=bias, pre_norm=False
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        '''
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if self.variant == 'original' and self.act is not None:
            out = self.act(out)

        return out
