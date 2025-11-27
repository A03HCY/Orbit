import numpy as np
from typing import Optional, Dict, Any, TYPE_CHECKING
from orbit.callback import Callback

if TYPE_CHECKING:
    from orbit.engine import Engine

class EarlyStopping(Callback):
    """
    Early Stopping 插件。
    如果监控的指标在 'patience' 个 Epoch 内没有改善，则停止训练。
    支持断点续训状态保存。
    """
    def __init__(
        self,
        monitor: str = 'val_loss',
        mode: str = 'min',
        patience: int = 5,
        min_delta: float = 0.0,
        verbose: bool = True
    ):
        """
        Args:
            monitor (str): 监控的指标名称 (e.g., 'val_loss', 'val_acc')。
            mode (str): 'min' (越小越好) 或 'max' (越大越好)。
            patience (int): 容忍多少个 Epoch 不提升。
            min_delta (float): 视为提升的最小变化量。
            verbose (bool): 是否打印信息。
        """
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        
        self.wait_count = 0
        self.best_score = np.inf if mode == 'min' else -np.inf
        
        # 内部状态 Key
        self._meta_key = 'early_stopping'

    def on_init(self, engine: 'Engine'):
        """尝试从 engine.meta 恢复状态"""
        if self._meta_key in engine.meta:
            state = engine.meta[self._meta_key]
            self.best_score = state.get('best_score', self.best_score)
            self.wait_count = state.get('wait_count', 0)
            if self.verbose:
                engine.print(f"[cyan]Resumed: Best Score={self.best_score:.4f}, Wait={self.wait_count}/{self.patience}[/]", plugin='EarlyStopping')

    def on_epoch_end(self, engine: 'Engine'):
        """每 Epoch 检查指标"""
        # 1. 获取当前指标
        current_score = engine.metrics.get(self.monitor)
        
        if current_score is None:
            # 如果指标不存在 (例如只跑了 Train 没跑 Eval)，跳过检查
            return

        # 2. 判断是否提升
        improved = False
        if self.mode == 'min':
            if current_score < self.best_score - self.min_delta:
                improved = True
        else:
            if current_score > self.best_score + self.min_delta:
                improved = True
        
        # 3. 更新状态
        if improved:
            old_best = self.best_score
            self.best_score = current_score
            self.wait_count = 0
            if self.verbose:
                if old_best == np.inf or old_best == -np.inf:
                    engine.print(f"{self.monitor} improved to [green]{current_score:.4f}[/]", plugin='EarlyStopping')
                else:
                    engine.print(f"{self.monitor} improved [green]{old_best:.4f} -> {current_score:.4f}[/]", plugin='EarlyStopping')
        else:
            self.wait_count += 1
            if self.verbose:
                engine.print(f"[yellow]{self.monitor} did not improve ({self.wait_count}/{self.patience}). Best: {self.best_score:.4f}[/]", plugin='EarlyStopping')
                
            if self.wait_count >= self.patience:
                engine.stop_training = True
                engine.print(f"[red][bold]Stopping training (no improvement in {self.patience} epochs).[/]", plugin='EarlyStopping')

        # 4. 保存状态到 meta，以便 Checkpoint 持久化
        engine.meta[self._meta_key] = {
            'best_score': self.best_score,
            'wait_count': self.wait_count
        }
