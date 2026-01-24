import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple

from orbit.model import BaseBlock, register_model

@dataclass
class QuantizerOutput:
    z_q: torch.Tensor
    loss: torch.Tensor
    indices: torch.Tensor
    entropy: torch.Tensor
    perplexity: torch.Tensor

@register_model()
class LFQ(BaseBlock):
    '''
    Lookup-Free Quantization (LFQ) 模块。
    
    基于 MagViT-2 论文。直接将 Latent 投影到低维空间进行二值化 (Sign)，
    并将二进制位组合成整数索引。
    
    优点：
    1. 计算效率高 (无最近邻搜索)。
    2. 支持超大词表 (如 codebook_dim=18 -> 262144 词表)。
    3. 训练更稳定。
    '''

    def __init__(
        self,
        latent_dim: int = 256,
        codebook_dim: int = 18,
        entropy_weight: float = 0.1,
        commitment_weight: float = 0.25,
        diversity_gamma: float = 1.0,
    ):
        '''
        Args:
            latent_dim (int): 输入/输出特征的维度 (Encoder 输出维度)。
            codebook_dim (int): 量化空间的维度 (Bit 数)。词表大小为 2^codebook_dim。
            entropy_weight (float): 熵损失权重，鼓励 Codebook 利用率。
            commitment_weight (float): 承诺损失权重，拉近 Encoder 输出与量化值的距离。
            diversity_gamma (float): 熵惩罚的缩放系数。
        '''
        super().__init__()
        self.latent_dim = latent_dim
        self.codebook_dim = codebook_dim
        self.entropy_weight = entropy_weight
        self.commitment_weight = commitment_weight
        self.diversity_gamma = diversity_gamma
        
        self.project_in = nn.Linear(latent_dim, codebook_dim) if latent_dim != codebook_dim else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, latent_dim) if latent_dim != codebook_dim else nn.Identity()
        
        self.register_buffer("basis", 2 ** torch.arange(codebook_dim))

    def entropy_loss(self, affine_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        计算基于 Bit 的熵损失。
        
        Args:
            affine_logits: 投影后的 logits [B*H*W, codebook_dim]
            
        Returns:
            loss: 熵损失标量 (希望熵最大化 -> 损失最小化)
            avg_entropy: 平均熵 (监控用)
        '''
        probs = torch.sigmoid(affine_logits)
        
        # [B*H*W, D] -> [D]
        avg_probs = torch.mean(probs, dim=0)
        
        entropy = - (avg_probs * torch.log(avg_probs + 1e-5) + 
                    (1 - avg_probs) * torch.log(1 - avg_probs + 1e-5))
        
        loss = - torch.mean(entropy) * self.diversity_gamma
        
        return loss, torch.mean(entropy)

    def forward(self, z: torch.Tensor) -> QuantizerOutput:
        '''
        Args:
            z (torch.Tensor): [B, C, H, W]
            
        Returns:
            QuantizerOutput
        '''
        B, C, H, W = z.shape
        
        z_permuted = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z_permuted.view(-1, C)
        
        z_e = self.project_in(z_flattened)
        
        z_q = torch.sign(z_e)
        
        z_q = z_e + (z_q - z_e).detach()
        
        commitment_loss = torch.mean((z_q.detach() - z_e) ** 2)
        
        entropy_loss, avg_entropy = self.entropy_loss(z_e)
        
        total_loss = self.commitment_weight * commitment_loss + self.entropy_weight * entropy_loss
        
        # [N, codebook_dim] * [codebook_dim] -> sum -> [N]
        is_positive = (z_q > 0).long()
        indices = (is_positive * self.basis).sum(dim=1)
        indices = indices.view(B, H, W)
        
        z_out = self.project_out(z_q)
        z_out = z_out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous() # [B, C, H, W]
        
        perplexity = 2 ** avg_entropy
        
        return QuantizerOutput(
            z_q=z_out,
            loss=total_loss,
            indices=indices,
            entropy=avg_entropy,
            perplexity=perplexity
        )
