from copy import deepcopy
import torch
from orbit.callback import Callback, Event
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING: from orbit.engine import Engine

class EMA(Callback):
    """
    指数移动平均 (Exponential Moving Average) 插件。
    在训练过程中维护模型参数的滑动平均版本，并在评估/预测时使用它。
    通常能提升模型的泛化能力和鲁棒性。
    """
    def __init__(self, decay: float = 0.999, start_step: int = 0):
        """
        Args:
            decay (float): 衰减率，通常接近 1 (如 0.999, 0.9999)。
            start_step (int): 从第几个 Global Step 开始启用 EMA。
        """
        super().__init__()
        self.decay = decay
        self.start_step = start_step
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        
        # 内部状态 Key，用于 Checkpoint 保存/恢复
        self._meta_key = 'ema_state'

    def on_init(self, event: Event):
        # 初始化影子权重 (Shadow Weights)
        # 注意：此时模型应该已经加载到了正确的 Device 上
        engine = event.engine
        for name, param in engine.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        
        engine.print(f"[magenta]Enabled (decay={self.decay})[/]", plugin='EMA')

    def on_train_start(self, event: Event):
        """尝试从 Checkpoint 恢复 EMA 状态"""
        engine = event.engine
        if self._meta_key in engine.meta:
            saved_shadow = engine.meta[self._meta_key]
            # 确保加载的权重在正确的设备上
            for k, v in saved_shadow.items():
                if k in self.shadow:
                    self.shadow[k] = v.to(engine.device)
            engine.print(f"[green]Resumed EMA state from checkpoint[/]", plugin='EMA')

    def on_batch_end(self, event: Event):
        """每个 Batch 结束后更新 EMA 权重"""
        engine = event.engine
        if engine.state == 'TRAIN' and engine.global_step >= self.start_step:
            for name, param in engine.model.named_parameters():
                if param.requires_grad:
                    # shadow = decay * shadow + (1 - decay) * param
                    self.shadow[name].data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def on_eval_start(self, event: Event):
        """评估开始前：备份当前权重，应用 EMA 权重"""
        engine = event.engine
        if engine.global_step < self.start_step:
            return

        self.backup = {
            name: p.data.clone() 
            for name, p in engine.model.named_parameters() 
            if p.requires_grad
        }
        
        for name, param in engine.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])
        
        engine.print("[dim]Switched to EMA weights for evaluation[/]", plugin='EMA')

    def on_eval_end(self, event: Event):
        """评估结束后：恢复原始训练权重"""
        engine = event.engine
        if not self.backup:
            return

        for name, param in engine.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        
        self.backup = {} # 清空备份
        engine.print("[dim]Restored training weights[/]", plugin='EMA')

    def on_epoch_end(self, event: Event):
        """
        Epoch 结束时：将 EMA 状态存入 meta，以便 Checkpoint 插件保存。
        注意：这会增加 Checkpoint 文件的大小 (约 2 倍模型大小)。
        """
        engine = event.engine
        engine.meta[self._meta_key] = self.shadow
