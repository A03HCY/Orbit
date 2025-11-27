from typing import TYPE_CHECKING
from orbit.callback import Callback

if TYPE_CHECKING:
    from orbit.engine import Engine

class GradientAccumulation(Callback):
    """
    梯度累积插件。
    通过配置 Engine 的 accumulation_steps 属性来实现。
    """
    def __init__(self, steps: int = 1):
        """
        Args:
            steps (int): 累积步数。默认为 1 (不累积)。
                         例如 steps=4，则每 4 个 Batch 更新一次参数，
                         等效 Batch Size = 原始 Batch Size * 4。
        """
        super().__init__()
        self.steps = steps
        
        if self.steps < 1:
            raise ValueError("Gradient accumulation steps must be >= 1")

    def on_init(self, engine: 'Engine'):
        """
        在初始化阶段配置 Engine
        """
        engine.accumulation_steps = self.steps
        if self.steps > 1:
            engine.print(f"[magenta]Enabled: steps={self.steps}[/]", plugin='GradAccum')
