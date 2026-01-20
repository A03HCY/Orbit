import torch.nn as nn
from typing import Literal
from orbit.utils.freeze import set_trainable

def set_moe_training_mode(
    model: nn.Module, 
    mode: Literal['all', 'router_only', 'experts_only'] = 'all',
    verbose: bool = True
) -> None:
    '''设置 MoE (Mixture of Experts) 模型的训练模式。
    
    通过识别模型中的 MoE 模块实例，精确控制 Router (门控网络) 和 Experts (专家网络) 的冻结/解冻状态。

    Args:
        model (nn.Module): 包含 MoE 结构的模型。
        mode (str): 训练模式。
            - 'all': 联合训练 Router 和 Experts (默认)。
            - 'router_only': 仅训练 Router，冻结 Experts。
            - 'experts_only': 仅训练 Experts，冻结 Router。
        verbose (bool): 是否打印状态变更信息。
    '''
    from orbit.model.block.moe import MoE

    moe_modules = [m for m in model.modules() if isinstance(m, MoE)]

    if not moe_modules:
        if verbose:
            print("Warning: No MoE modules found in the provided model.")
        return

    count = len(moe_modules)
    
    for moe in moe_modules:
        if mode == 'all':
            set_trainable(moe.router, trainable=True)
            set_trainable(moe.experts, trainable=True)
            
        elif mode == 'router_only':
            set_trainable(moe.experts, trainable=False)
            set_trainable(moe.router, trainable=True)
            
        elif mode == 'experts_only':
            set_trainable(moe.router, trainable=False)
            set_trainable(moe.experts, trainable=True)
            
        else:
            raise ValueError(f"Unknown mode: {mode}. Supported modes: 'all', 'router_only', 'experts_only'")

    if verbose:
        mode_desc = {
            'all': "ALL (Router: Unfrozen, Experts: Unfrozen)",
            'router_only': "ROUTER_ONLY (Router: Unfrozen, Experts: Frozen)",
            'experts_only': "EXPERTS_ONLY (Router: Frozen, Experts: Unfrozen)"
        }
        print(f"MoE Training Mode set to: {mode_desc[mode]} for {count} MoE module(s).")
