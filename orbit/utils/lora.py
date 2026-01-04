import torch
import torch.nn as nn
from typing import Union, Optional, List

from orbit.model import LinearLoRA, Conv2dLoRA, Conv1dLoRA, EmbeddingLoRA

lora_models = [LinearLoRA, Conv2dLoRA, Conv1dLoRA, EmbeddingLoRA]

def freeze_backbone_only(
    model: nn.Module, 
    unlock_head_keywords: list = None,
    verbose: bool = True
):
    '''冻结骨干网络，仅保留 LoRA 层和指定的头部层可训练。

    该函数首先冻结所有参数，然后解冻 LoRA 模块（LinearLoRA, Conv2dLoRA）的参数，
    最后解冻名称中包含 unlock_head_keywords 中任意关键字的参数。

    Args:
        model (nn.Module): 目标模型。
        unlock_head_keywords (list, optional): 需要保持解冻状态的头部层关键字列表。
            例如 ['head', 'fc', 'classifier']。默认为 None。
        verbose (bool): 是否打印冻结状态统计信息。默认为 True。
    '''
    for param in model.parameters():
        param.requires_grad = False
    
    if not unlock_head_keywords: unlock_head_keywords = []
        
    lora_types = tuple(lora_models)
    
    lora_counter = 0
    for name, module in model.named_modules():
        if isinstance(module, lora_types):
            lora_counter += 1
            
            if hasattr(module, 'lora_a') and module.lora_a is not None:
                if isinstance(module.lora_a, nn.Module):
                    for p in module.lora_a.parameters(): p.requires_grad = True
                elif isinstance(module.lora_a, nn.Parameter):
                    module.lora_a.requires_grad = True
                    
            if hasattr(module, 'lora_b') and module.lora_b is not None:
                if isinstance(module.lora_b, nn.Module):
                    for p in module.lora_b.parameters(): p.requires_grad = True
                elif isinstance(module.lora_b, nn.Parameter):
                    module.lora_b.requires_grad = True
            
            if hasattr(module, 'dora_m') and module.dora_m is not None:
                module.dora_m.requires_grad = True
                
            if hasattr(module, 'lora_gate') and module.lora_gate is not None:
                module.lora_gate.requires_grad = True
                    
        for name, param in model.named_parameters():
            if any(k in name for k in unlock_head_keywords):
                param.requires_grad = True
                
    if verbose:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Backbone frozen.")
        print(f"- Active LoRA blocks: {lora_counter}")
        print(f"- Trainable params: {trainable:,} / {total:,} ({trainable/total:.2%})")
        if unlock_head_keywords:
            print(f"- Extra unlocked layers: {unlock_head_keywords}")


def inject_lora(
    model: nn.Module, 
    r: int = 8, 
    lora_alpha: int = 16, 
    lora_dropout: float = 0.05,
    gate: bool = False,
    dora: bool = False,
    target_names: list = None,
    exclude_names: list = None
):
    '''向模型中注入 LoRA 层。

    递归遍历模型，将 Linear, Conv2d, Conv1d, Embedding 层替换为对应的 LoRA 包装层。
    支持普通 LoRA、Gated LoRA 和 DoRA。

    Args:
        model (nn.Module): 目标模型。
        r (int): LoRA 的秩。默认为 8。
        lora_alpha (int): LoRA 的缩放系数。默认为 16。
        lora_dropout (float): Dropout 概率。默认为 0.05。
        gate (bool): 是否启用 Gated LoRA (添加可学习的门控参数)。默认为 False。
        dora (bool): 是否启用 DoRA (Weight-Decomposed Low-Rank Adaptation)。默认为 False。
        target_names (list, optional): 仅注入名称包含这些关键字的层。默认为 None (注入所有支持的层)。
        exclude_names (list, optional): 排除名称包含这些关键字的层。默认为 None。

    Returns:
        nn.Module: 注入 LoRA 后的模型。
    '''
    for name, child in model.named_children():
        is_target = target_names is None or any(t in name for t in target_names)
        is_excluded = exclude_names is not None and any(e in name for e in exclude_names)
        
        if not is_target or is_excluded:
            inject_lora(child, r, lora_alpha, lora_dropout, gate, dora, target_names, exclude_names)
            continue
            
        if isinstance(child, nn.Linear):
            new_layer = LinearLoRA(
                child, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, 
                gate=gate, dora=dora
            )
            setattr(model, name, new_layer)
            
        elif isinstance(child, nn.Conv2d):
            if child.kernel_size == (1, 1) or child.kernel_size == 1:
                pass
            else:
                new_layer = Conv2dLoRA(
                    child, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                    gate=gate, dora=dora
                )
                setattr(model, name, new_layer)

        elif isinstance(child, nn.Conv1d):
            new_layer = Conv1dLoRA(
                child, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                gate=gate, dora=dora
            )
            setattr(model, name, new_layer)

        elif isinstance(child, nn.Embedding):
            new_layer = EmbeddingLoRA(
                child, r=r, lora_alpha=lora_alpha,
                gate=gate, dora=dora
            )
            setattr(model, name, new_layer)
                
        else:
            inject_lora(child, r, lora_alpha, lora_dropout, gate, dora, target_names, exclude_names)
            
    return model


def save_lora(model: nn.Module, path: str):
    '''仅保存模型的 LoRA 权重。

    遍历模型的 state_dict，提取所有键名中包含 'lora_' 的参数并保存。

    Args:
        model (nn.Module): 包含 LoRA 层的模型。
        path (str): 保存路径。
    '''
    lora_state_dict = {}
    full_state_dict = model.state_dict()
    
    for key, value in full_state_dict.items():
        if 'lora_' in key:
            lora_state_dict[key] = value
            
    torch.save(lora_state_dict, path)
    print(f"LoRA weights saved to {path}. Size: {len(lora_state_dict)} keys.")

def load_lora(model: nn.Module, path: str):
    '''加载 LoRA 权重到模型中。

    使用 strict=False 加载权重，并打印缺失或意外的键的警告信息。
    支持加载纯权重文件或 Checkpoint 插件保存的包含 'model_state_dict' 的字典。

    Args:
        model (nn.Module): 目标模型。
        path (str): 权重文件路径。
    '''
    checkpoint = torch.load(path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        lora_state_dict = checkpoint['model_state_dict']
    else:
        lora_state_dict = checkpoint

    missing, unexpected = model.load_state_dict(lora_state_dict, strict=False)
    
    if unexpected:
        print(f"Warning: Unexpected keys found: {unexpected}")
    
    lora_missing = [k for k in missing if 'lora_' in k]
    if lora_missing:
        print(f"Warning: Missing LoRA keys: {lora_missing}")
    else:
        print("LoRA weights loaded successfully.")

class LoRADiagnoser:
    @staticmethod
    def get_status(model: nn.Module, verbose: bool = False) -> dict:
        """
        在 train loop 中调用此函数，返回当前 LoRA 层的健康状态。
        """
        stats = {
            "total_lora_modules": 0,
            "active_grads": 0,
            "avg_update_magnitude": 0.0,
            "max_grad_norm": 0.0,
            "dead_neurons": 0
        }
        
        update_ratios = []
        
        for name, module in model.named_modules():
            if hasattr(module, 'lora_a') and hasattr(module, 'lora_b') and module.r > 0:
                stats["total_lora_modules"] += 1
                
                wa = module.lora_a.weight if isinstance(module.lora_a, nn.Module) else module.lora_a
                wb = module.lora_b.weight if isinstance(module.lora_b, nn.Module) else module.lora_b
                
                if wa.grad is not None and wb.grad is not None:
                    stats["active_grads"] += 1
                    g_norm = wa.grad.norm().item() + wb.grad.norm().item()
                    stats["max_grad_norm"] = max(stats["max_grad_norm"], g_norm)
                
                s = module.scaling
                
                norm_a = wa.data.norm().item()
                norm_b = wb.data.norm().item()
                norm_delta = norm_a * norm_b * s
                
                if hasattr(module, 'original_layer'):
                    # Conv LoRA
                    norm_w = module.original_layer.weight.data.norm().item()
                elif hasattr(module, 'weight'):
                    norm_w = module.weight.data.norm().item()
                else:
                    norm_w = 1.0 # Fallback
                
                ratio = norm_delta / (norm_w + 1e-6)
                update_ratios.append(ratio)
                
                if norm_b < 1e-9:
                    stats["dead_neurons"] += 1

        if update_ratios:
            stats["avg_update_magnitude"] = sum(update_ratios) / len(update_ratios)
            stats["min_ratio"] = min(update_ratios)
            stats["max_ratio"] = max(update_ratios)
            
        if verbose:
            print(f"--- LoRA Diagnosis ---")
            print(f"Modules: {stats['total_lora_modules']} | Active Grads: {stats['active_grads']}")
            print(f"Update Ratio (Perturbation): {stats['avg_update_magnitude']:.6f} (Target ~0.001 - 0.01)")
            print(f"Max Gradient Norm: {stats['max_grad_norm']:.6f}")
            if stats['dead_neurons'] > 0:
                print(f"Warning: {stats['dead_neurons']} modules have near-zero output (Initialization issue?)")
            print(f"----------------------")
            
        return stats

    @staticmethod
    def check_collapse(model: nn.Module, threshold: float = 1e-4):
        """
        检查 LoRA 矩阵是否存在严重的秩塌缩 (Rank Collapse)。
        """
        print("Running SVD analysis on LoRA layers...")
        for name, module in model.named_modules():
            if hasattr(module, 'lora_a') and hasattr(module, 'lora_b'):
                wa = module.lora_a.weight if isinstance(module.lora_a, nn.Module) else module.lora_a
                
                if wa.dim() > 2:
                    wa_flat = wa.view(wa.shape[0], -1) # (r, in*k*k)
                else:
                    wa_flat = wa
                    
                try:
                    _, S, _ = torch.svd(wa_flat)
                    # 归一化奇异值
                    S = S / S[0]
                    effective_rank = (S > threshold).sum().item()
                    print(f"[{name}] Rank: {module.r} | Effective Rank: {effective_rank} | Top/Bottom Ratio: {S[0]/S[-1]:.1f}")
                except:
                    pass
