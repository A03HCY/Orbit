from typing import Optional, List, TYPE_CHECKING
from orbit.callback import Callback, Event

if TYPE_CHECKING: from orbit.engine import Engine

class Warmup(Callback):
    """
    学习率预热 (Warmup) 插件。
    支持 Linear, Constant, Noam (Transformer) 三种模式。
    可以在 Batch 粒度上动态调整学习率。
    """
    def __init__(
        self,
        warmup_steps: int = 0,
        warmup_epochs: int = 0,
        mode: str = 'linear',
        min_lr: float = 0.0,
        model_dim: Optional[int] = None,
        scale: float = 1.0,
    ):
        """
        Args:
            warmup_steps (int): 预热的总步数 (Batch 数)。
            warmup_epochs (int): 预热的总 Epoch 数。如果设置，优先级高于 warmup_steps。
            mode (str): 'linear' | 'constant' | 'noam'。
            min_lr (float): 起始学习率 (linear/constant 模式)。
            model_dim (int): 模型维度 (仅 noam 模式需要)。
            scale (float): 缩放因子 (仅 noam 模式需要)。
        """
        super().__init__()
        self.warmup_steps = warmup_steps
        self.warmup_epochs = warmup_epochs
        self.mode = mode.lower()
        self.min_lr = min_lr
        self.model_dim = model_dim
        self.scale = scale

        self.total_warmup_steps = 0
        self.base_lrs: List[float] = []
        
        # 验证参数
        if self.mode == 'noam' and self.model_dim is None:
            raise ValueError("Noam mode requires 'model_dim' to be specified.")

    def on_train_start(self, event: Event):
        """
        训练开始时计算总预热步数并记录初始学习率
        """
        engine = event.engine
        if not engine.optimizer:
            raise ValueError("Warmup plugin requires an optimizer in the Engine.")

        # 1. 记录优化器的初始学习率 (Base LR)
        # 如果是从 Checkpoint 恢复，base_lrs 可能会变，但在 Warmup 逻辑里我们通常认为
        # 预热的目标就是 param_group['initial_lr'] (如果存在) 或者当前的 ['lr']
        self.base_lrs = []
        for group in engine.optimizer.param_groups:
            # 优先使用 initial_lr (由某些 Scheduler 设置)，否则使用当前 lr
            self.base_lrs.append(group.get('initial_lr', group['lr']))

        # 2. 计算 total_warmup_steps
        if self.warmup_epochs > 0:
            if not engine.train_loader:
                raise ValueError("warmup_epochs requires train_loader to be available.")
            try:
                steps_per_epoch = len(engine.train_loader)
                self.total_warmup_steps = self.warmup_epochs * steps_per_epoch
            except TypeError:
                 # 如果 train_loader 无法求 len (例如 iterable dataset)，则必须提供 steps
                 raise ValueError("Could not determine length of train_loader. Please use 'warmup_steps' instead.")
        else:
            self.total_warmup_steps = self.warmup_steps

        # 打印信息
        if self.total_warmup_steps > 0 or self.mode == 'noam':
            engine.print(f"[magenta]Strategy activated: {self.mode}[/]", plugin='Warmup')
            if self.mode != 'noam':
                engine.print(f"[magenta]Steps: {self.total_warmup_steps} (Epochs: {self.warmup_epochs})[/]", plugin='Warmup')

    def on_batch_start(self, event: Event):
        """
        每个 Batch 开始前调整学习率
        """
        engine = event.engine
        # 当前步数 (从 1 开始计算，方便公式)
        current_step = engine.batch_idx + 1
        
        # 如果超出预热范围且不是 Noam 模式，则不进行干预
        # Noam 模式是一种全程调度策略，所以它会一直运行
        if self.mode != 'noam' and current_step > self.total_warmup_steps:
            return

        for i, group in enumerate(engine.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            new_lr = base_lr

            if self.mode == 'noam':
                # Noam Scheduler: scale * d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)
                warmup = self.total_warmup_steps
                # 避免 step=0
                step = max(1, current_step)
                
                term1 = step ** -0.5
                term2 = step * (warmup ** -1.5)
                
                new_lr = self.scale * (self.model_dim ** -0.5) * min(term1, term2)
                
            elif self.mode == 'linear':
                # Linear: min_lr -> base_lr
                alpha = current_step / self.total_warmup_steps
                new_lr = self.min_lr + (base_lr - self.min_lr) * alpha
                
            elif self.mode == 'constant':
                # Constant: min_lr
                new_lr = self.min_lr
            
            # 更新学习率
            group['lr'] = new_lr
