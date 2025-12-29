from orbit.callback import Callback, Event
from typing import Any

class Overfit(Callback):
    '''故意重复第一个 Batch 的数据以进行过拟合测试的插件。
    
    这对于验证模型架构是否有能力拟合数据非常有用（Sanity Check）。
    如果模型无法在单个 Batch 上过拟合，则可能存在代码错误或架构问题。
    '''
    
    def __init__(self):
        self.fixed_data: Any = None
        self.fixed_target: Any = None
        self.has_captured = False

    def on_batch_start(self, event: Event):
        # 仅在训练阶段生效
        if event.engine.state != "TRAIN":
            return

        if not self.has_captured:
            # 捕获第一个 Batch
            self.fixed_data = event.engine.data
            self.fixed_target = event.engine.target
            self.has_captured = True
            event.engine.print("[yellow]Overfit Plugin: Captured first batch. All subsequent batches will be replaced by this one.[/]", plugin='Overfit')
        else:
            # 替换后续 Batch
            event.engine.data = self.fixed_data
            event.engine.target = self.fixed_target
