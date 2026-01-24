import torch
import torch.nn as nn
import torch.nn.functional as F
from lpips import LPIPS
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Union

@dataclass
class VQGANGeneratorLog:
    ''' VQGAN 生成器阶段的日志数据类。
    
    Attributes:
        total_loss (torch.Tensor): 总损失。
        quant_loss (torch.Tensor): 量化损失。
        nll_loss (torch.Tensor): 负对数似然损失（重构损失）。
        p_loss (torch.Tensor): 感知损失。
        rec_loss (torch.Tensor): 总重构损失（像素 + 感知）。
        d_weight (torch.Tensor): 对抗损失的自适应权重。
        g_loss (torch.Tensor): 生成器对抗损失。
    '''
    total_loss: torch.Tensor
    quant_loss: torch.Tensor
    nll_loss: torch.Tensor
    p_loss: torch.Tensor
    rec_loss: torch.Tensor
    d_weight: torch.Tensor
    g_loss: torch.Tensor

@dataclass
class VQGANDiscriminatorLog:
    ''' VQGAN 判别器阶段的日志数据类。
    
    Attributes:
        disc_loss (torch.Tensor): 判别器总损失。
        logits_real (torch.Tensor): 真实样本的 logits 均值。
        logits_fake (torch.Tensor): 生成样本的 logits 均值。
    '''
    disc_loss: torch.Tensor
    logits_real: torch.Tensor
    logits_fake: torch.Tensor

@dataclass
class VQGANLossOutput:
    ''' VQGANLoss 的输出数据类。
    
    Attributes:
        loss (torch.Tensor): 总损失标量。
        log (Union[VQGANGeneratorLog, VQGANDiscriminatorLog]): 损失日志对象。
    '''
    loss: torch.Tensor
    log: Union[VQGANGeneratorLog, VQGANDiscriminatorLog]

class VQGANLoss(nn.Module):
    ''' VQGAN 模型的损失函数模块。
    
    结合了感知损失 (LPIPS)、重构损失 (L1/L2)、对抗损失 (GAN Loss) 和代码本损失。
    '''
    def __init__(
            self, 
            disc_start: int = 10000,
            kl_weight: float = 1.0,
            pixelloss_weight: float = 1.0, 
            perceptual_weight: float = 1.0,
            disc_weight: float = 0.8,
            disc_factor: float = 1.0
        ):
        ''' 初始化 VQGANLoss。

        Args:
            disc_start (int): 判别器开始训练的步数。默认为 10000。
            logvar_init (float): 对数方差的初始值。默认为 0.0。
            kl_weight (float): KL 散度损失的权重。默认为 1.0。
            pixelloss_weight (float): 像素级重构损失的权重。默认为 1.0。
            perceptual_weight (float): 感知损失的权重。默认为 1.0。
            disc_weight (float): 判别器损失的权重。默认为 0.8。
            disc_factor (float): 判别器损失的缩放因子。默认为 1.0。
        '''
        super().__init__()
        
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        self.disc_factor = disc_factor
        self.disc_weight = disc_weight
        self.disc_start = disc_start
        
        self.perceptual_loss = LPIPS(net='vgg', verbose=False).eval()
        
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer_weights):
        ''' 计算自适应对抗损失权重 lambda。

        Args:
            nll_loss (torch.Tensor): 负对数似然损失（重构损失）。
            g_loss (torch.Tensor): 生成器损失。
            last_layer_weights (torch.Tensor): 解码器最后一层的权重。

        Returns:
            torch.Tensor: 自适应权重。
        '''
        nll_grads = torch.autograd.grad(nll_loss, last_layer_weights, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer_weights, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight * self.disc_weight

    def forward(
            self, 
            inputs: torch.Tensor,
            reconstructions: torch.Tensor,
            quantizer_loss: torch.Tensor,
            global_step: int,
            last_layer_weights: torch.Tensor,
            discriminator: nn.Module,
            optimizer_idx: int,
            mask: torch.Tensor = None
        ) -> VQGANLossOutput:
        ''' 前向计算损失。

        Args:
            inputs (torch.Tensor): 原始输入图像。
            reconstructions (torch.Tensor): 重建图像。
            quantizer_loss (torch.Tensor): 量化器损失。
            global_step (int): 当前全局步数。
            last_layer_weights (torch.Tensor): 解码器最后一层的权重。
            discriminator (nn.Module): 判别器模型。
            optimizer_idx (int): 优化器索引（0 为生成器，1 为判别器）。
            mask (torch.Tensor, optional): 有效区域掩码。默认为 None。

        Returns:
            VQGANLossOutput: 包含 loss 和 log 的对象。
        '''
        rec_loss_tensor = torch.abs(inputs - reconstructions) 

        if mask is not None:
            if mask.shape[-2:] != rec_loss_tensor.shape[-2:]:
                mask = F.interpolate(mask, size=rec_loss_tensor.shape[-2:], mode='nearest')
            
            mask_expanded = mask.expand_as(rec_loss_tensor)
            nll_loss = (rec_loss_tensor * mask_expanded).sum() / (mask_expanded.sum() + 1e-6)
        else:
            nll_loss = torch.mean(rec_loss_tensor)

        p_loss_scalar = torch.tensor(0.0, device=inputs.device)
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs, reconstructions)
            p_loss_scalar = p_loss.mean()
        
        rec_loss_total = nll_loss * self.pixel_weight + self.perceptual_weight * p_loss_scalar

        if optimizer_idx == 0:
            logits_fake = discriminator(reconstructions)
            g_loss = -torch.mean(logits_fake)
            
            try: d_weight = self.calculate_adaptive_weight(rec_loss_total, g_loss, last_layer_weights)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = 1 if global_step >= self.disc_start else 0
            
            loss = rec_loss_total + \
                   self.kl_weight * quantizer_loss + \
                   d_weight * disc_factor * g_loss
                   
            log = VQGANGeneratorLog(
                total_loss=loss.detach(),
                quant_loss=quantizer_loss.detach(),
                nll_loss=nll_loss.detach(),
                p_loss=p_loss_scalar.detach(),
                rec_loss=rec_loss_total.detach(),
                d_weight=d_weight.detach(),
                g_loss=g_loss.detach()
            )
            return VQGANLossOutput(loss=loss, log=log)

        if optimizer_idx == 1:
            logits_real = discriminator(inputs.detach())
            logits_fake = discriminator(reconstructions.detach())
            
            disc_factor = 1 if global_step >= self.disc_start else 0
            
            # Hinge Loss
            loss_real = torch.mean(F.relu(1. - logits_real))
            loss_fake = torch.mean(F.relu(1. + logits_fake))
            d_loss = disc_factor * 0.5 * (loss_real + loss_fake)

            log = VQGANDiscriminatorLog(
                disc_loss=d_loss.detach(),
                logits_real=logits_real.mean().detach(),
                logits_fake=logits_fake.mean().detach()
            )
            return VQGANLossOutput(loss=d_loss, log=log)
