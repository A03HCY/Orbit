import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Tuple, Optional

from orbit.model import BaseBlock, register_model
from orbit.utils.image import pad_to_patch_size
from orbit.model.block import (
    ConvBlock, ResBasicBlock,
    SpatialMultiHeadAttention, AttentionOutput,
    MRoPEInterleavedEmbedding,
    LFQ, QuantizerOutput
)

def dynamic_collate_fn(batch):
    '''
    将不同尺寸的图片 Batch 整理为统一尺寸 (Padding)，并生成有效区域掩码。
    
    Returns:
        padded_batch: [B, C, Max_H, Max_W]
        mask_batch:   [B, 1, Max_H, Max_W] (1.0 = Valid, 0.0 = Padding)
    '''
    images = [item for item in batch] 
    
    max_h = max([img.shape[1] for img in images])
    max_w = max([img.shape[2] for img in images])
    
    stride = 16 # Patch Size
    max_h = ((max_h + stride - 1) // stride) * stride
    max_w = ((max_w + stride - 1) // stride) * stride
    
    batch_size = len(images)
    channels = images[0].shape[0]
    
    padded_batch = torch.zeros(batch_size, channels, max_h, max_w)
    mask_batch = torch.zeros(batch_size, 1, max_h, max_w)
    
    for i, img in enumerate(images):
        h, w = img.shape[1], img.shape[2]
        padded_batch[i, :, :h, :w] = img
        mask_batch[i, :, :h, :w] = 1.0
        
    return padded_batch, mask_batch

@dataclass
class EncoderOutput:
    '''
    VQGAN Encoder 的输出数据类。

    Attributes:
        output (torch.Tensor): 编码后的潜在变量，形状为 [B, out_channels, h_latent, w_latent]。
        mask (torch.Tensor): 原始分辨率的有效区域掩码，形状为 [B, 1, H_pad, W_pad]。
        input_shape (Tuple[int, int]): 原始输入图像的尺寸 (H, W)。
        padded_shape (Tuple[int, int]): Padding 后的图像尺寸 (H_pad, W_pad)。
    '''
    output: torch.Tensor
    mask: torch.Tensor
    input_shape: Tuple[int, int]
    padded_shape: Tuple[int, int]

@dataclass
class DecoderOutput:
    '''
    VQGAN Decoder 的输出数据类。
    
    Attributes:
        reconstruction (torch.Tensor): 重建的图像 [B, 3, H, W]。
    '''
    reconstruction: torch.Tensor

@dataclass
class MotifV1Info:
    '''
    MotifV1 的辅助信息数据类。
    
    Attributes:
        perplexity (torch.Tensor): 码本困惑度。
        entropy (torch.Tensor): 码本熵。
        indices (torch.Tensor): 量化索引 [B, H, W]。
        mask (Optional[torch.Tensor]): 有效区域掩码 [B, 1, H, W]。
    '''
    perplexity: torch.Tensor
    entropy: torch.Tensor
    indices: torch.Tensor
    mask: Optional[torch.Tensor]

@dataclass
class MotifV1Output:
    '''
    MotifV1 主模型的输出数据类。
    
    Attributes:
        reconstruction (torch.Tensor): 重建的图像 [B, 3, H, W]。
        loss (torch.Tensor): 量化损失 (Commitment Loss)。
        info (MotifV1Info): 辅助信息对象。
    '''
    reconstruction: torch.Tensor
    loss: torch.Tensor
    info: MotifV1Info
    
@register_model()
class MotifV1Encoder(BaseBlock):
    '''
    支持可变输入尺寸和 2D RoPE 的 VQGAN Encoder。

    该编码器通过一系列下采样卷积块和残差块提取特征，并在中间层应用空间多头注意力机制（Spatial Multi-Head Attention）
    和二维旋转位置编码（2D RoPE）。它能够处理任意尺寸的输入图像，通过 padding 确保尺寸满足 patch_size 的要求。
    '''
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 128,
        out_channels: int = 256,
        patch_size: int = 16,
        num_res_blocks: int = 1,
        num_heads: int = 4,
        dropout: float = 0.0,
        rope_max_len: int = 4096,
        max_channels: int = 256
    ):
        '''
        初始化 MotifV1Encoder

        Args:
            in_channels (int): 输入图像的通道数。默认为 3。
            hidden_dim (int): 隐藏层的初始维度。默认为 128。
            out_channels (int): 输出潜在变量的通道数。默认为 256。
            patch_size (int): Patch 大小，必须是 2 的幂。用于计算下采样次数。默认为 16。
            num_res_blocks (int): 每个下采样阶段的残差块数量。默认为 1。
            num_heads (int): 注意力机制的头数。默认为 4。
            dropout (float): Dropout 概率。默认为 0.0。
            rope_max_len (int): RoPE 的最大长度限制。默认为 4096。
            max_channels (int): 隐藏层通道数的最大值。默认为 256。

        Raises:
            ValueError: 如果 patch_size 不是 2 的幂。
        '''
        super().__init__()
        
        if (patch_size & (patch_size - 1)) != 0:
            raise ValueError(f"patch_size must be a power of 2, got {patch_size}")
        
        self.patch_size = patch_size
        self.num_downsamples = int(torch.log2(torch.tensor(patch_size)))
        
        self.conv_in = ConvBlock(
            in_channels, hidden_dim, kernel_size=3, padding=1, norm=None, activation=None
        )

        self.down_blocks = nn.ModuleList()
        curr_dim = hidden_dim
        
        for _ in range(self.num_downsamples):
            blocks = []
            for _ in range(num_res_blocks):
                blocks.append(ResBasicBlock(
                    curr_dim,
                    curr_dim,
                    dropout=dropout,
                    norm='group',
                    activation='silu',
                    variant='pre_act'
                ))
            self.down_blocks.append(nn.Sequential(*blocks))
            
            next_dim = min(curr_dim * 2, max_channels)
            
            self.down_blocks.append(ConvBlock(
                curr_dim, next_dim, kernel_size=3, stride=2, padding=1, 
                norm=None, activation=None
            ))
            curr_dim = next_dim

        self.mid_res1 = ResBasicBlock(
            curr_dim,
            curr_dim,
            dropout=dropout,
            norm='group',
            activation='silu',
            variant='pre_act'
        )
        
        self.mid_attn = SpatialMultiHeadAttention(
            hidden_size=curr_dim,
            num_heads=num_heads,
            use_qk_norm=True
        )
        
        self.rope = MRoPEInterleavedEmbedding(
            model_dim=curr_dim // num_heads,
            num_axes=2, 
            max_len=rope_max_len
        )

        self.mid_res2 = ResBasicBlock(
            curr_dim,
            curr_dim,
            dropout=dropout,
            norm='group',
            activation='silu',
            variant='pre_act'
        )

        self.norm_out = nn.GroupNorm(32, curr_dim)
        self.act_out = nn.SiLU(inplace=True)
        self.conv_out = nn.Conv2d(curr_dim, out_channels, kernel_size=3, padding=1)

    def _build_grid(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        '''
        构建 2D 网格坐标。

        Args:
            h (int): 网格的高度。
            w (int): 网格的宽度。
            device (torch.device): 张量所在的设备。

        Returns:
            torch.Tensor: 网格坐标张量，形状为 [h * w, 2]。
        '''
        y = torch.arange(h, device=device)
        x = torch.arange(w, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([grid_y, grid_x], dim=-1).reshape(-1, 2)
        return grid

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        '''
        前向传播函数。

        Args:
            x (torch.Tensor): 输入图像张量，形状为 [B, C, H, W]。

        Returns:
            EncoderOutput: 包含编码输出、掩码和尺寸信息的对象。
        '''
        B, C, H, W = x.shape
        
        patch_out = pad_to_patch_size(x, (self.patch_size, self.patch_size))
        x_in = patch_out.output
        mask = patch_out.mask 
        
        h = self.conv_in(x_in)
        
        for layer in self.down_blocks:
            h = self.checkpoint(layer, h)
        
        h = self.checkpoint(self.mid_res1, h)
        
        B_feat, C_feat, H_feat, W_feat = h.shape
        h_flat = h.permute(0, 2, 3, 1).reshape(B_feat, H_feat * W_feat, C_feat)
        
        positions = self._build_grid(H_feat, W_feat, h.device)
        positions = positions.unsqueeze(0).expand(B_feat, -1, -1)
        
        attn_out: AttentionOutput = self.checkpoint(
            self.mid_attn,
            hidden_states=h_flat,
            positions=positions,
            rotary_emb=self.rope
        )
        
        h = attn_out.output.view(B_feat, H_feat, W_feat, C_feat).permute(0, 3, 1, 2)
        h = self.checkpoint(self.mid_res2, h)
        
        h = self.norm_out(h)
        h = self.act_out(h)
        z = self.conv_out(h)
        
        return EncoderOutput(
            output=z,
            mask=mask, 
            input_shape=(H, W),
            padded_shape=patch_out.output.shape[-2:]
        )

@register_model()
class MotifV1Decoder(BaseBlock):
    '''
    基于 PixelShuffle 的 VQGAN Decoder。
    
    结构：Input Conv -> Mid Block -> Upsample Stack (PixelShuffle) -> Output Conv
    支持根据 Encoder 提供的掩码自动裁剪输出图像。
    '''
    def __init__(
        self,
        in_channels: int = 256,
        hidden_dim: int = 256,
        out_channels: int = 3,
        patch_size: int = 16,
        num_res_blocks: int = 1,
        num_heads: int = 4,
        dropout: float = 0.0,
        rope_max_len: int = 4096,
    ):
        '''
        初始化 MotifV1Decoder

        Args:
            in_channels (int): 输入潜在变量的通道数。默认为 256。
            hidden_dim (int): 隐藏层的初始维度。默认为 256。
            out_channels (int): 输出图像的通道数。默认为 3。
            patch_size (int): Patch 大小，必须是 2 的幂。用于计算上采样次数。默认为 16。
            num_res_blocks (int): 每个上采样阶段的残差块数量。默认为 1。
            num_heads (int): 注意力机制的头数。默认为 4。
            dropout (float): Dropout 概率。默认为 0.0。
            rope_max_len (int): RoPE 的最大长度限制。默认为 4096。

        Raises:
            ValueError: 如果 patch_size 不是 2 的幂。
        '''
        super().__init__()
        
        if (patch_size & (patch_size - 1)) != 0:
            raise ValueError(f"patch_size must be a power of 2, got {patch_size}")

        self.patch_size = patch_size
        self.num_upsamples = int(torch.log2(torch.tensor(patch_size)))
        
        self.conv_in = ConvBlock(
            in_channels, hidden_dim, kernel_size=3, padding=1, norm=None, activation=None
        )

        self.mid_res1 = ResBasicBlock(
            hidden_dim,
            hidden_dim,
            dropout=dropout,
            norm='group',
            activation='silu',
            variant='pre_act'
        )
        
        self.mid_attn = SpatialMultiHeadAttention(
            hidden_size=hidden_dim,
            num_heads=num_heads,
            use_qk_norm=True
        )
        
        self.rope = MRoPEInterleavedEmbedding(
            model_dim=hidden_dim // num_heads,
            num_axes=2, 
            max_len=rope_max_len
        )
        
        self.mid_res2 = ResBasicBlock(
            hidden_dim,
            hidden_dim,
            dropout=dropout,
            norm='group',
            activation='silu',
            variant='pre_act'
        )
        
        self.up_blocks = nn.ModuleList()
        curr_dim = hidden_dim
        
        for _ in range(self.num_upsamples):
            blocks = []
            for _ in range(num_res_blocks):
                blocks.append(ResBasicBlock(
                    curr_dim,
                    curr_dim,
                    dropout=dropout,
                    norm='group',
                    activation='silu',
                    variant='pre_act'
                ))
            self.up_blocks.append(nn.Sequential(*blocks))
            
            next_dim = curr_dim // 2
            
            if next_dim < 64: next_dim = 64
            
            self.up_blocks.append(nn.Sequential(
                ConvBlock(curr_dim, next_dim * 4, kernel_size=3, padding=1, norm='group', activation='silu'),
                nn.PixelShuffle(2)
            ))
            curr_dim = next_dim

        self.norm_out = nn.GroupNorm(32, curr_dim)
        self.act_out = nn.SiLU(inplace=True)
        self.conv_out = nn.Conv2d(curr_dim, out_channels, kernel_size=3, padding=1)

    def _build_grid(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        '''
        构建 2D 网格坐标。

        Args:
            h (int): 网格的高度。
            w (int): 网格的宽度。
            device (torch.device): 张量所在的设备。

        Returns:
            torch.Tensor: 网格坐标张量，形状为 [h * w, 2]。
        '''
        y = torch.arange(h, device=device)
        x = torch.arange(w, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([grid_y, grid_x], dim=-1).reshape(-1, 2)
        return grid

    def forward(self, z: torch.Tensor, mask: torch.Tensor = None) -> DecoderOutput:
        '''
        Args:
            z (torch.Tensor): 量化后的 Latent [B, C, H, W]
            mask (torch.Tensor, optional): Encoder 输出的掩码 [B, 1, H_orig, W_orig]。
                                           如果提供，将用于裁剪 Padding。

        Returns:
            DecoderOutput: 包含重建图像的对象。
        '''
        B, C, H, W = z.shape
        
        h = self.conv_in(z)
        
        h = self.checkpoint(self.mid_res1, h)
        
        B_feat, C_feat, H_feat, W_feat = h.shape
        h_flat = h.permute(0, 2, 3, 1).reshape(B_feat, H_feat * W_feat, C_feat)
        
        positions = self._build_grid(H_feat, W_feat, h.device)
        positions = positions.unsqueeze(0).expand(B_feat, -1, -1)
        
        attn_out = self.checkpoint(
            self.mid_attn,
            hidden_states=h_flat,
            positions=positions,
            rotary_emb=self.rope
        )
        
        h = attn_out.output.view(B_feat, H_feat, W_feat, C_feat).permute(0, 3, 1, 2)
        h = self.checkpoint(self.mid_res2, h)
        
        for layer in self.up_blocks:
            h = self.checkpoint(layer, h)
            
        h = self.norm_out(h)
        h = self.act_out(h)
        recon = self.conv_out(h) # [B, 3, H_pad, W_pad]
            
        return DecoderOutput(reconstruction=recon)

@register_model()
class MotifV1(BaseBlock):
    '''
    Motif-V1 主模型。
    整合 Encoder, Decoder 和 LFQ Quantizer。
    '''
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_dim: int = 256,
        codebook_dim: int = 18,
        patch_size: int = 16,
        
        enc_hidden_dim: int = 64,
        enc_num_res_blocks: int = 1,
        enc_num_heads: int = 4,
        enc_max_channels: int = 256,
        
        dec_hidden_dim: int = 256,
        dec_num_res_blocks: int = 1,
        dec_num_heads: int = 4,
        
        dropout: float = 0.0,
        rope_max_len: int = 4096,
        
        entropy_weight: float = 0.1,
        commitment_weight: float = 0.25,
    ):
        '''
        初始化 MotifV1 主模型。

        Args:
            in_channels (int): 输入图像的通道数。默认为 3。
            out_channels (int): 输出图像的通道数。默认为 3。
            latent_dim (int): Encoder 输出和 Decoder 输入的潜在变量维度。默认为 256。
            codebook_dim (int): LFQ 码本的比特数（维度）。默认为 18。
            patch_size (int): Patch 大小，必须是 2 的幂。默认为 16。
            enc_hidden_dim (int): Encoder 隐藏层的初始维度。默认为 64。
            enc_num_res_blocks (int): Encoder 每个阶段的残差块数量。默认为 1。
            enc_num_heads (int): Encoder 注意力机制的头数。默认为 4。
            enc_max_channels (int): Encoder 隐藏层通道数的最大值。默认为 256。
            dec_hidden_dim (int): Decoder 隐藏层的初始维度。默认为 256。
            dec_num_res_blocks (int): Decoder 每个阶段的残差块数量。默认为 1。
            dec_num_heads (int): Decoder 注意力机制的头数。默认为 4。
            dropout (float): Dropout 概率。默认为 0.0。
            rope_max_len (int): RoPE 的最大长度限制。默认为 4096。
            entropy_weight (float): LFQ 熵损失的权重。默认为 0.1。
            commitment_weight (float): LFQ 承诺损失的权重。默认为 0.25。
        '''
        super().__init__()
        
        self.codebook_dim = codebook_dim
        
        self.encoder = MotifV1Encoder(
            in_channels=in_channels,
            hidden_dim=enc_hidden_dim,
            out_channels=latent_dim,
            patch_size=patch_size,
            num_res_blocks=enc_num_res_blocks,
            num_heads=enc_num_heads,
            dropout=dropout,
            rope_max_len=rope_max_len,
            max_channels=enc_max_channels
        )
        
        self.quantizer = LFQ(
            latent_dim=latent_dim,
            codebook_dim=codebook_dim,
            entropy_weight=entropy_weight,
            commitment_weight=commitment_weight
        )
        
        self.decoder = MotifV1Decoder(
            in_channels=latent_dim,
            hidden_dim=dec_hidden_dim,
            out_channels=out_channels,
            patch_size=patch_size,
            num_res_blocks=dec_num_res_blocks,
            num_heads=dec_num_heads,
            dropout=dropout,
            rope_max_len=rope_max_len
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        '''
        初始化模型权重。

        Args:
            m (nn.Module): 需要初始化的模块。
        
        Note:
            - Conv2d, Linear, Conv1d 使用 Xavier Uniform 初始化，偏置为 0。
            - LayerNorm, GroupNorm, BatchNorm2d 权重为 1，偏置为 0。
        '''
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    @property
    def last_layer_weights(self):
        '''用于 VQGANLoss 计算自适应权重'''
        return self.decoder.conv_out.weight

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        仅编码：将图像编码为量化索引。

        Args:
            x (torch.Tensor): 输入图像，形状为 [B, C, H, W]。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - indices (torch.Tensor): 量化索引，形状为 [B, H, W]。
                - mask (torch.Tensor): 有效区域掩码，形状为 [B, 1, H, W]。
                - z_q (torch.Tensor): 量化后的潜在变量（用于调试），形状为 [B, C, H, W]。
        '''
        enc_out = self.encoder(x)
        q_out = self.quantizer(enc_out.output)
        return q_out.indices, enc_out.mask, q_out.z_q

    def decode(self, indices: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        '''
        仅解码：将量化索引解码为图像（通常用于 Transformer 推理）。

        Args:
            indices (torch.Tensor): 量化索引，形状为 [B, H, W]。
            mask (torch.Tensor, optional): 有效区域掩码，用于裁剪输出。默认为 None。

        Returns:
            torch.Tensor: 重建的图像，形状为 [B, 3, H, W]。
        '''
        z_q = self.indices_to_codes(indices) # [B, H, W, Codebook_Dim]
        
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        if hasattr(self.quantizer, 'project_out'):
            z_q_permuted = z_q.permute(0, 2, 3, 1)
            z_projected = self.quantizer.project_out(z_q_permuted)
            # [B, H, W, Latent_Dim] -> [B, Latent_Dim, H, W]
            z_q = z_projected.permute(0, 3, 1, 2).contiguous()

        dec_out = self.decoder(z_q, mask)
        return dec_out.reconstruction

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        '''
        LFQ 核心工具：将整数索引还原为二值化向量。
        
        将整数索引转换为二进制码（-1 或 1）。

        Args:
            indices (torch.Tensor): 量化索引，形状为 [B, H, W]。

        Returns:
            torch.Tensor: 二值化向量，形状为 [B, H, W, Codebook_Dim]，值为 -1.0 或 1.0。
        '''
        B, H, W = indices.shape
        basis = self.quantizer.basis # [Codebook_Dim]
        
        basis_long = basis.long()
        indices_long = indices.long().unsqueeze(-1)

        is_one = (indices_long & basis_long) > 0 
        
        codes = torch.where(is_one, torch.tensor(1.0, device=indices.device), torch.tensor(-1.0, device=indices.device))
        
        return codes

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> MotifV1Output:
        '''
        前向传播函数。

        Args:
            x (torch.Tensor): 已经 Pad 过的输入图像，形状为 [B, 3, H, W]。
            mask (torch.Tensor, optional): 外部传入的有效区域掩码（通常由 collate_fn 生成），形状为 [B, 1, H, W]。
                                           如果为 None，则使用 Encoder 生成的掩码。

        Returns:
            MotifV1Output: 包含重建图像、损失和辅助信息的对象。
        '''
        enc_out: EncoderOutput = self.encoder(x)
        
        q_out: QuantizerOutput = self.quantizer(enc_out.output)
        
        dec_out: DecoderOutput = self.decoder(q_out.z_q, mask)
        
        final_mask = mask if mask is not None else enc_out.mask
        info = MotifV1Info(
            perplexity=q_out.perplexity,
            entropy=q_out.entropy,
            indices=q_out.indices,
            mask=final_mask
        )
        
        return MotifV1Output(
            reconstruction=dec_out.reconstruction,
            loss=q_out.loss,
            info=info
        )
