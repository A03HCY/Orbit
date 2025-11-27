import torch
import torch.nn as nn
from rich.panel import Panel
from rich.table import Table
from rich import box
from typing import TYPE_CHECKING, Optional

from orbit.callback import Callback

if TYPE_CHECKING:
    from orbit.engine import Engine

class MemoryEstimator(Callback):
    """
    显存预估插件。
    在训练开始前，通过运行一个虚拟 Batch 来预估显存使用峰值。
    """
    def __init__(self, verbose: bool = True):
        super().__init__()
        self.verbose = verbose
        self.has_run = False

    def on_train_start(self, engine: 'Engine'):
        if self.has_run:
            return

        if not torch.cuda.is_available():
            if self.verbose:
                engine.print("[yellow]CUDA not available. Skipping memory estimation.[/]", plugin='MemEst')
            return

        # 确保模型在正确的设备上
        device = engine.device
        if device.type != 'cuda':
            return

        try:
            self._estimate(engine)
        except Exception as e:
            engine.print(f"[red]Error during memory estimation: {e}[/]", plugin='MemEst')
        finally:
            # 清理
            if engine.optimizer:
                engine.optimizer.zero_grad()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.has_run = True

    def _estimate(self, engine: 'Engine'):
        engine.print("[cyan]Running dry run for memory estimation...[/]", plugin='MemEst')
        
        # 1. 获取一个 Batch 的数据
        try:
            batch_data = next(iter(engine.train_loader))
        except StopIteration:
            engine.print("[yellow]Train loader is empty. Skipping.[/]", plugin='MemEst')
            return

        # 2. 准备环境
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
        
        # 计算模型静态大小 (Weights + Buffers)
        model_stats = self._get_model_size(engine.model)
        
        # 3. 模拟 Forward & Backward
        try:
            # 移动数据
            engine._process_batch_data(batch_data)
            
            # Forward
            with torch.amp.autocast(device_type=engine.device.type, enabled=engine.use_amp):
                if isinstance(engine.data, (list, tuple)):
                    output = engine.model(*engine.data)
                else:
                    output = engine.model(engine.data)
                
                # 构造虚拟 Loss
                if engine.criterion and engine.target is not None:
                    loss = engine.criterion(output, engine.target)
                else:
                    # 如果没有 target 或 criterion，构造一个标量 loss 用于 backward
                    if isinstance(output, torch.Tensor):
                        loss = output.mean()
                    elif isinstance(output, (list, tuple)) and isinstance(output[0], torch.Tensor):
                        loss = output[0].mean()
                    elif isinstance(output, dict):
                        loss = list(output.values())[0].mean()
                    else:
                        loss = torch.tensor(0.0, device=engine.device, requires_grad=True)

            # Backward
            if engine.use_amp and engine.scaler:
                engine.scaler.scale(loss).backward()
            else:
                loss.backward()

            # 获取峰值显存
            peak_memory = torch.cuda.max_memory_allocated()
            total_capacity = torch.cuda.get_device_properties(engine.device).total_memory
            
            self._print_report(engine, model_stats, initial_memory, peak_memory, total_capacity)

        except RuntimeError as e:
            if "out of memory" in str(e):
                engine.print("[bold red]OOM detected during memory estimation![/]", plugin='MemEst')
                engine.print(f"[red]Your batch size is likely too large for this device.[/]", plugin='MemEst')
            else:
                raise e

    def _get_model_size(self, model: nn.Module) -> float:
        """计算模型参数和缓冲区的总字节数"""
        mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
        mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
        return mem_params + mem_bufs

    def _print_report(self, engine: 'Engine', model_size: int, initial: int, peak: int, capacity: int):
        if not self.verbose: return

        # 转换单位为 MB
        to_mb = lambda x: x / (1024 ** 2)
        
        model_mb = to_mb(model_size)
        peak_mb = to_mb(peak)
        capacity_mb = to_mb(capacity)
        usage_percent = (peak / capacity) * 100
        
        # 颜色编码
        if usage_percent < 70:
            color = "green"
            status = "Safe"
        elif usage_percent < 90:
            color = "yellow"
            status = "Warning"
        else:
            color = "red"
            status = "Critical"

        table = Table(box=box.SIMPLE, show_header=False)
        table.add_column("Item", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Model Weights", f"{model_mb:.2f} MB")
        table.add_row("Est. Peak Memory", f"[{color}]{peak_mb:.2f} MB[/]")
        table.add_row("Device Capacity", f"{capacity_mb:.2f} MB")
        table.add_row("Usage", f"[{color}]{usage_percent:.1f}% ({status})[/]")
        
        panel = Panel(
            table,
            title="[bold]Memory Estimation Report[/]",
            border_style="blue",
            expand=False
        )
        
        with engine.out_logs:
            engine.console.print(panel)
