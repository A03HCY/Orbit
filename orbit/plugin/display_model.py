import torch
import torch.nn as nn
from rich.table import Table
from rich.console import Console
from typing import TYPE_CHECKING
import rich.box as box

from orbit.callback import Callback

if TYPE_CHECKING:
    from ..engine import Engine

class ModelSummary(Callback):
    def __init__(self, max_depth: int = 3):
        super().__init__()
        self.max_depth = max_depth

    def on_init(self, engine: "Engine"):
        """
        Engine 初始化时，自动打印模型结构
        """
        self.display(engine.model, engine.console)

    def display(self, model: nn.Module, console: Console):
        """核心打印逻辑"""
        table = Table(title=f"[bold]Model Summary: {model.__class__.__name__}[/]", box=box.HORIZONTALS)
        
        table.add_column("Layer (Type)", style="cyan", no_wrap=True)
        table.add_column("Output Shape", style="magenta")
        table.add_column("Param #", justify="right", style="green")
        table.add_column("Trainable", justify="right", style="yellow")

        total_params = 0
        trainable_params = 0
        
        # 遍历顶层模块 (简单版遍历，深度遍历比较复杂，为了美观这里展示第一级子模块)
        for name, module in model.named_children():
            # 计算该模块的总参数
            num_params = sum(p.numel() for p in module.parameters())
            num_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            total_params += num_params
            trainable_params += num_trainable
            
            is_trainable = "[bold green]Yes[/]" if num_trainable > 0 else "[dim]No[/]"
            
            # 尝试获取输出形状 (这通常需要 hook, 这里为了简单我们只显示 Layer Type)
            # 想要准确的 output shape 需要像 torchinfo/torchsummary 那样做一次 fake forward
            layer_name = f"{name} ({module.__class__.__name__})"
            
            table.add_row(
                layer_name, 
                "-", # 若不做 forward 推理，很难静态获取 output shape
                f"{num_params:,}", 
                is_trainable
            )

        # 计算模型总大小 (MB) - Float32 = 4 bytes
        total_size_mb = total_params * 4 / (1024 ** 2)

        console.print(table)
        
        # 打印汇总信息
        summary_table = Table(show_header=False, box=None)
        summary_table.add_row("Total Params:", f"[bold cyan]{total_params:,}[/]")
        summary_table.add_row("Trainable Params:", f"[bold green]{trainable_params:,}[/] ({trainable_params/total_params:.1%})")
        summary_table.add_row("Non-trainable Params:", f"[dim]{total_params - trainable_params:,}[/]")
        summary_table.add_row("Est. Params Size (MB):", f"[bold blue]{total_size_mb:.2f} MB[/]")
        
        console.print(summary_table)
        console.print(' ' + '─' * 15 + '┬' + '─' * 35)