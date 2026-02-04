import math
from typing import List
from orbit.callback import Callback, Event
from orbit.plugin.warmup import Warmup

class Annealing(Callback):
    """
    学习率退火 (Annealing) 插件。
    通常在 Warmup 结束后开始，将学习率从 Peak LR 逐渐衰减到 Min LR。
    支持 Cosine 和 Linear 两种模式。
    """
    def __init__(self, mode: str = 'cosine', min_lr: float = 0.0):
        """
        Args:
            mode (str): 'cosine' | 'linear'。默认为 'cosine'。
            min_lr (float): 退火结束时的最低学习率。
        """
        super().__init__()
        self.mode = mode.lower()
        self.min_lr = min_lr
        
        self.warmup_steps = 0
        self.total_train_steps = 0
        self.base_lrs: List[float] = []

    def on_train_start(self, event: Event):
        engine = event.engine
        
        if not engine.optimizer:
            raise ValueError("Annealing plugin requires an optimizer in the Engine.")

        # 1. 尝试自动检测 Warmup 步数
        # 遍历已挂载的插件，寻找 Warmup 实例
        for p in engine.plugins:
            if isinstance(p, Warmup):
                self.warmup_steps = p.total_warmup_steps
                break
        
        # 2. 计算总训练步数
        # 依赖 engine.train_loader 和 engine.num_epochs
        if engine.train_loader:
            try:
                # 尝试获取 loader 长度
                steps_per_epoch = len(engine.train_loader)
                self.total_train_steps = engine.num_epochs * steps_per_epoch
            except (TypeError, AttributeError):
                # 如果无法获取长度 (例如 IterableDataset)，且没有手动提供信息，则无法进行基于 step 的退火
                engine.print("[yellow]Warning: Could not determine total training steps from train_loader. Annealing might not work as expected.[/]", plugin='Annealing')
                self.total_train_steps = 0
        else:
             # 如果没有 train_loader (例如手动循环)，可能需要其他方式获取 steps，这里暂不处理
             pass

        # 3. 记录初始 Base LR (即 Peak LR)
        self.base_lrs = []
        for group in engine.optimizer.param_groups:
            # 优先使用 initial_lr (通常由 Scheduler 或之前的逻辑设定为 Peak LR)
            # 如果没有 initial_lr，则使用当前的 lr
            self.base_lrs.append(group.get('initial_lr', group['lr']))

        # 打印信息
        if self.total_train_steps > 0:
            annealing_steps = max(0, self.total_train_steps - self.warmup_steps)
            engine.print(f"[magenta]Strategy activated: {self.mode}[/]", plugin='Annealing')
            engine.print(f"[magenta]Annealing Steps: {annealing_steps} (Total: {self.total_train_steps}, Warmup: {self.warmup_steps})[/]", plugin='Annealing')

    def on_batch_start(self, event: Event):
        if self.total_train_steps <= 0:
            return

        engine = event.engine
        current_step = engine.global_step + 1
        
        # 如果还在 Warmup 阶段，不进行退火
        if current_step <= self.warmup_steps:
            return
        
        # 如果超出总步数，保持 min_lr
        if current_step > self.total_train_steps:
            for group in engine.optimizer.param_groups:
                group['lr'] = self.min_lr
            return

        # 计算退火进度 (0.0 -> 1.0)
        annealing_step = current_step - self.warmup_steps
        total_annealing_steps = max(1, self.total_train_steps - self.warmup_steps)
        progress = annealing_step / total_annealing_steps
        
        # Clamp progress to [0, 1] just in case
        progress = min(max(progress, 0.0), 1.0)

        for i, group in enumerate(engine.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            new_lr = base_lr

            if self.mode == 'cosine':
                # Cosine Annealing
                # lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
                decay = 0.5 * (1 + math.cos(math.pi * progress))
                new_lr = self.min_lr + (base_lr - self.min_lr) * decay
            
            elif self.mode == 'linear':
                # Linear Annealing
                # lr = min_lr + (base_lr - min_lr) * (1 - progress)
                decay = 1.0 - progress
                new_lr = self.min_lr + (base_lr - self.min_lr) * decay
            
            group['lr'] = new_lr
