import gc
import torch
import torch.nn as nn
from rich.panel import Panel
from rich.table import Table
from rich import box
from typing import TYPE_CHECKING, Optional, Union

from orbit.callback import Callback, Event

if TYPE_CHECKING: from orbit.engine import Engine

class MemoryEstimator(Callback):
    """
    显存预估插件。
    在训练开始前，通过运行一个虚拟 Batch 来预估显存使用峰值。
    同时支持在训练过程中监控显存使用情况。
    """
    def __init__(self, verbose: bool = True, alert_threshold: Union[float, str, int] = 0.8, stop_threshold: Union[float, str, int] = 0.95, clean_interval: Optional[int] = None):
        '''
        Args:
            verbose (bool): 是否打印预估报告。
            alert_threshold (Union[float, str, int]): 警告阈值。
                如果是 float <= 1.0，视为总显存的百分比 (例如 0.8 = 80%)。
                如果是 str (例如 "4GB", "500MB") 或 int (字节数)，视为绝对值。
            stop_threshold (Union[float, str, int]): 停止阈值。类型同上。
            clean_interval (Optional[int]): 如果提供，则每隔多少个 Batch 执行一次显存清理 (gc.collect + empty_cache)。
                这有助于解决显存缓慢泄漏的问题，但可能会轻微影响训练速度。
        '''
        super().__init__()
        self.verbose = verbose
        self.alert_threshold_arg = alert_threshold
        self.stop_threshold_arg = stop_threshold
        self.clean_interval = clean_interval
        
        self.alert_bytes = None
        self.stop_bytes = None
        
        self.has_run = False
        self.has_alerted = False

    def _parse_threshold(self, threshold: Union[float, str, int], total_capacity: int) -> int:
        if isinstance(threshold, float):
            return int(threshold * total_capacity)
        if isinstance(threshold, int):
            return threshold
        if isinstance(threshold, str):
            s = threshold.upper().strip()
            if s.endswith('GB'):
                return int(float(s[:-2]) * (1024**3))
            elif s.endswith('MB'):
                return int(float(s[:-2]) * (1024**2))
            elif s.endswith('KB'):
                return int(float(s[:-2]) * 1024)
            elif s.endswith('B'):
                return int(float(s[:-1]))
            else:
                try:
                    return int(float(s))
                except ValueError:
                    pass
        raise ValueError(f"Invalid memory threshold format: {threshold}")

    def on_batch_start(self, event: Event):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_batch_end(self, event: Event):
        if not torch.cuda.is_available():
            return
        
        engine = event.engine
        peak_memory = torch.cuda.max_memory_allocated()
        total_capacity = torch.cuda.get_device_properties(engine.device).total_memory
        
        # 初始化阈值字节数
        if self.stop_bytes is None:
            self.stop_bytes = self._parse_threshold(self.stop_threshold_arg, total_capacity)
        if self.alert_bytes is None:
            self.alert_bytes = self._parse_threshold(self.alert_threshold_arg, total_capacity)
        
        # 格式化辅助函数
        to_mb = lambda x: x / (1024 ** 2)
        
        if peak_memory > self.stop_bytes:
            engine.print(f"[bold red]Memory usage ({to_mb(peak_memory):.2f} MB) exceeded critical threshold ({to_mb(self.stop_bytes):.2f} MB)! Stopping training.[/]", plugin='MemEst')
            engine.stop(source="MemoryEstimator", reason=f"Memory usage exceeded critical threshold ({to_mb(self.stop_bytes):.2f} MB)")
        elif peak_memory > self.alert_bytes and not self.has_alerted:
            engine.print(f"[yellow]Memory usage ({to_mb(peak_memory):.2f} MB) exceeded warning threshold ({to_mb(self.alert_bytes):.2f} MB).[/]", plugin='MemEst')
            self.has_alerted = True

        # 定期清理显存
        if self.clean_interval and (engine.batch_idx + 1) % self.clean_interval == 0:
            gc.collect()
            torch.cuda.empty_cache()

    def on_train_start(self, event: Event):
        if self.has_run:
            return
        
        engine = event.engine
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
        if self.verbose:
            engine.print("Running dry run for memory estimation...[/]", plugin='MemEst')
        
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
