import torch
import torch.nn as nn
from typing import List, Optional

from orbit.model import BaseBlock, register_model
from orbit.model.block.conv import CausalConv1d

@register_model()
class TCN(BaseBlock):
    '''
    时间卷积网络 (Temporal Convolutional Network, TCN)。
    
    由一系列因果空洞卷积层 (Causal Dilated Convolutions) 组成。
    支持手动指定每层通道数或根据目标感受野自动构建。
    '''

    def __init__(
        self,
        in_channels: int,
        num_channels: Optional[List[int]] = None,
        out_channels: Optional[int] = None,
        step: Optional[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        use_res: bool = True,
        norm: str = None,
        activation: str = 'leaky_relu',
        leaky_relu: float = 0.1
    ):
        '''
        初始化 TCN 模块。

        Args:
            in_channels (int): 输入通道数。
            num_channels (List[int], optional): 每一层的输出通道数列表。如果提供此参数，将忽略 out_channels 和 step。
            out_channels (int, optional): 自动构建模式下的统一输出通道数。
            step (int, optional): 自动构建模式下的目标感受野 (时间步长)。
            kernel_size (int, optional): 卷积核大小。默认为 3。
            dropout (float, optional): Dropout 概率。默认为 0.2。
            use_res (bool, optional): 是否使用残差连接。默认为 True。
            norm (str, optional): 归一化类型 (传递给 CausalConv1d/ConvBlock)。默认为 None。
            activation (str, optional): 激活函数类型。默认为 'leaky_relu'。
            leaky_relu (float, optional): LeakyReLU 的负斜率。默认为 0.1。
        '''
        super(TCN, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        
        if num_channels is not None:
            layers = []
            num_levels = len(num_channels)
            for i in range(num_levels):
                dilation_size = 2 ** i
                in_ch = in_channels if i == 0 else num_channels[i-1]
                out_ch = num_channels[i]
                
                layers.append(CausalConv1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation_size,
                    norm=norm,
                    activation=activation,
                    leaky_relu=leaky_relu,
                    use_res=use_res,
                    dropout=dropout
                ))
            self.network = nn.Sequential(*layers)
            self.out_channels = num_channels[-1]
            
        elif step is not None and out_channels is not None:
            self.network = CausalConv1d.auto_block(
                in_channels=in_channels,
                out_channels=out_channels,
                step=step,
                kernel_size=kernel_size,
                norm=norm,
                activation=activation,
                leaky_relu=leaky_relu,
                use_res=use_res,
                dropout=dropout
            )
            self.out_channels = out_channels
        else:
            raise ValueError("Must provide either 'num_channels' (list) or both 'step' and 'out_channels' (int).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        前向传播。

        Args:
            x (torch.Tensor): 输入张量。Shape: [Batch, in_channels, Seq_Len]

        Returns:
            torch.Tensor: 输出张量。Shape: [Batch, out_channels, Seq_Len]
        '''
        return self.network(x)
