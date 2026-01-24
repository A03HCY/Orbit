from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional
from dataclasses import dataclass
import torch

if TYPE_CHECKING: from .engine import Engine

@dataclass
class Event:
    '''事件数据类，携带当前 Engine 状态及触发源信息。'''
    engine: Engine
    name: str
    source: Optional[str] = None
    reason: Optional[str] = None

class Callback:
    """
    回调基类。
    所有方法都接收 event 实例，允许修改 engine 状态或读取数据。
    """
    def on_init(self, event: Event): ...
    
    def on_train_start(self, event: Event): ...
    def on_train_end(self, event: Event): ...
    
    def on_epoch_start(self, event: Event): ...
    def on_epoch_end(self, event: Event): ...
    
    def on_batch_start(self, event: Event): ...
    def on_batch_end(self, event: Event): ...
    
    def on_eval_start(self, event: Event): ...
    def on_eval_end(self, event: Event): ...

    def on_requested_stop(self, event: Event): ...
    def on_exception(self, event: Event): ...

class Forward:
    '''自定义前向传播和 Loss 计算接口。

    实现此接口以接管 Engine 的默认前向传播逻辑。
    '''

    def forward(self, engine: Engine, data: Any, target: Any) -> torch.Tensor:
        '''执行前向传播并返回 Loss。

        Args:
            engine (Engine): 当前 Engine 实例。
            data (Any): 当前 Batch 的输入数据。
            target (Any): 当前 Batch 的目标数据（标签）。

        Returns:
            torch.Tensor: 计算得到的 Loss 标量。
        '''
        ... 
