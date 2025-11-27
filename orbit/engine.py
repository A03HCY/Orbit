import torch
import torch.nn as nn
from typing import Any, List, Optional, Union, Dict, Tuple

try:
    from torch.utils.tensorboard import SummaryWriter
except: pass

from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.console import Console

from .callback import Callback 
from orbit.plugin.checkpoint import Checkpoint
from orbit.plugin.board import Board
from orbit.plugin.display_model import ModelSummary

class Engine:

    class _OutLogs:
        def __init__(self, engine: 'Engine'):
            self.engine = engine
        def __enter__(self):
            self.engine._print_edge(top=False)
            self.engine.console.print('\n')
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.engine.console.print('\n')
            self.engine._print_edge(top=True)

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer = None,
        criterion: nn.Module = None,
        device: Optional[str] = None,
        use_amp: bool = False,
        grad_clip_norm: float = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        plugins: List[Callback] = None,
        checkpoint_dir: str = None,
        console: Console = None,
    ):
        # --- 基础组件 ---
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model_name = self.model.__class__.__name__
        self.optimizer = optimizer
        self.criterion = criterion
        
        # --- 训练配置 ---
        self.use_amp = use_amp
        self.grad_clip_norm = grad_clip_norm
        self.scheduler = scheduler
        self.scaler = torch.amp.GradScaler('cuda', enabled=use_amp) 

        # --- 交互与回调 ---
        self.console = console if console else Console()
        self.out_logs = self._OutLogs(self)
        self.writer: Optional[SummaryWriter] = None
        self.plugins = [
            ModelSummary(model),
            Checkpoint(name=self.model_name, path=checkpoint_dir) if checkpoint_dir else None,
        ]
        self.plugins = [p for p in self.plugins if p is not None]
        self.attach(plugins)

        # --- 内部状态 (State) ---
        self.num_epochs = 0
        self.start_epoch = 0
        
        self.global_step = 0     # 全局 Step
        self.epoch = 0           # 当前 Epoch
        self.batch_idx = 0       # 当前 Batch 索引
        
        self.state = "IDLE"      # TRAIN / EVAL
        self.stop_training = False # 插件可以通过设置此标志为 True 来停止训练
        self.accumulation_steps = 1 # 梯度累积步数

        self.exception: Optional[Exception] = None
        
        # 当前 Batch 的数据容器
        self.data: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]] = None
        self.target: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]] = None
        self.output: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]] = None
        self.loss: torch.Tensor = None
        self.metrics: Dict[str, Any] = {} # 存放每个Epoch的统计指标

        # --- 持久化元数据 (Meta) ---
        # 这是一个随 Checkpoint 保存和加载的字典。
        # 插件可以使用这个字典来存储任何需要在训练中断/恢复后保持的状态。
        # 例如：EarlyStopping 的 best_score, Warmup 的状态等。
        # 使用方法: engine.meta['plugin_name'] = { ... state ... }
        self.meta: Dict[str, Any] = {}

        # 触发初始化回调
        self._fire_event("on_init")
    
    def init_board(self, log_dir: str = 'runs') -> 'Engine':
        board = Board(name=self.model_name, log_dir=log_dir)
        board.on_init(self)
        self.attach(board)
        return self

    def set_checkpoint(self, dir: str, name: Optional[str] = None, **kwargs) -> 'Engine':
        """
        配置 Checkpoint 插件。
        如果已存在 Checkpoint 插件，将被替换。
        
        Args:
            dir (str): 保存目录。
            name (str): 模型名称前缀。如果为 None，则使用 model_name。
            **kwargs: 传递给 Checkpoint 构造函数的其他参数 (monitor, save_top_k 等)。
        """
        if name is None:
            name = self.model_name
            
        # 1. 移除旧的 Checkpoint 插件 (如果存在)
        self.plugins = [p for p in self.plugins if not isinstance(p, Checkpoint)]
        
        # 2. 创建新插件
        ckpt = Checkpoint(name=name, path=dir, **kwargs)
        
        # 3. 调用 ckpt.on_init(self)
        ckpt.on_init(self)
        
        # 4. 挂载
        self.attach(ckpt)
        return self
    
    def _print_edge(self, top=True):
        char = '┬' if top else '┴'
        self.console.print(' ' + '─' * 15 + char + '─' * 35)
    
    def print(self, *args, plugin: Optional[str] = None, **kwargs):
        """
        统一打印方法。
        Args:
            plugin (str): 插件名称。如果提供，将以固定宽度对齐打印。
        """
        if plugin:
            # 宽度 15, 右对齐, 青色加粗
            prefix = f"[bold cyan]{plugin:>15}[/] │"
            self.console.print(prefix, *args, **kwargs)
        else:
            self.console.print(*args, **kwargs)
    
    def attach(self, plugin: Union[Callback, List[Callback]]=None):
        if not plugin: return
        if isinstance(plugin, Callback):
            plugin = [plugin]
        for p in plugin:
            if not isinstance(p, Callback):
                raise ValueError(f"Plugin {p} is not a Callback!")
            if p in self.plugins: continue
            self.plugins.append(p)

    def _fire_event(self, event_name: str):
        """触发所有 Callback 的对应方法"""
        for cb in self.plugins:
            method = getattr(cb, event_name, None)
            if method:
                # [修改] 移除 try-except pass。
                # 我们需要看到 Callback 里的错误，否则调试是地狱。
                # 如果一定要防御性编程，可以使用 console.print_exception()
                method(self) 

    def _process_batch_data(self, batch_data: Any):
        """将数据移动到设备"""
        if isinstance(batch_data, (list, tuple)):
            batch_data = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch_data]
            if len(batch_data) == 2:
                self.data, self.target = batch_data
            elif len(batch_data) == 1:
                self.data = batch_data[0]
                self.target = None
            else:
                self.data = batch_data[:-1]
                self.target = batch_data[-1]
        elif isinstance(batch_data, dict):
            self.data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch_data.items()}
            self.target = None 
        else:
            self.data = batch_data.to(self.device)
            self.target = None

    def run(self, train_loader, val_loader=None, num_epochs=10, start_epoch=None, with_eval=True):
        """主要的入口方法"""
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        # 优先应该使用传入的 start_epoch，如果没有则维持原状 (支持 Checkpoint 修改)
        if start_epoch is not None:
            self.start_epoch = start_epoch

        self._fire_event("on_train_start")
        try:
            for epoch in range(self.start_epoch, self.num_epochs):
                self.epoch = epoch
                
                # --- 1. Training Loop ---
                self.state = "TRAIN"
                self._fire_event("on_epoch_start")
                self._run_one_epoch(self.train_loader, prefix="Train", color="blue")

                # --- 2. Validation Loop ---
                if self.val_loader and with_eval:
                    self.state = "EVAL"
                    self._fire_event("on_eval_start")
                    with torch.no_grad():
                        self._run_one_epoch(self.val_loader, prefix="Eval ", color="yellow")
                    self._fire_event("on_eval_end")

                if self.scheduler:
                    self.scheduler.step()

                self._fire_event("on_epoch_end")
                
                if self.stop_training:
                    self.print("[yellow]Training stopped by plugin request.[/]", plugin='Engine')
                    self._fire_event("on_requested_stop")
                    break
                    
        except KeyboardInterrupt:
            self.print("[red][bold]Training interrupted by user.", plugin='Engine')
            self._fire_event("on_requested_stop")
        except Exception as e:
            self.exception = e
            self.console.print_exception()
            self._fire_event("on_exception")
        finally:
            self._fire_event("on_train_end")
            self._print_edge(top=False)

    def _run_one_epoch(self, loader, prefix="Train", color="blue"):
        is_train = (self.state == "TRAIN")
        self.model.train() if is_train else self.model.eval()
        
        epoch_loss_sum = 0.0
        num_batches = len(loader)
        
        with Progress(
            TextColumn(f"[{color}]{prefix}"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=True
        ) as progress:
            
            task = progress.add_task(f"[Ep {self.epoch+1}/{self.num_epochs}]", total=num_batches)
            
            for batch_idx, batch_data in enumerate(loader):
                self.batch_idx = batch_idx
                
                self._process_batch_data(batch_data)
                self._fire_event("on_batch_start")

                # --- Forward ---
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    if isinstance(self.data, (list, tuple)):
                        self.output = self.model(*self.data)
                    else:
                        self.output = self.model(self.data)

                    if self.output is None:
                        raise ValueError("Model returned None! Please check your model's forward() method.")
                    
                    if self.criterion and self.target is not None:
                        self.target = self.target
                        self.loss = self.criterion(self.output, self.target)
                    else:
                        self.loss = torch.tensor(0.0, device=self.device)
                    
                    loss_val = self.loss.item()
                    epoch_loss_sum += loss_val

                # --- Backward (仅训练模式) ---
                if is_train:
                    # 1. 梯度累积：Loss 缩放
                    if self.accumulation_steps > 1:
                        self.loss = self.loss / self.accumulation_steps
                    
                    # 2. Backward (计算梯度)
                    if self.use_amp and self.scaler:
                        self.scaler.scale(self.loss).backward()
                    else:
                        self.loss.backward()

                    # 3. Optimizer Step (仅在累积步数到达或 Epoch 结束时执行)
                    if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                        if self.use_amp and self.scaler:
                            if self.grad_clip_norm:
                                self.scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            if self.grad_clip_norm:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                            self.optimizer.step()
                        
                        self.optimizer.zero_grad()
                        self.global_step += 1

                # 更新进度条
                logs = f"Loss: {loss_val:.4f} [Ep {self.epoch+1}/{self.num_epochs}]"
                progress.update(task, advance=1, description=logs)
                
                self._fire_event("on_batch_end")
        
        # 计算 epoch 平均 loss
        avg_loss = epoch_loss_sum / num_batches if num_batches > 0 else 0.0
        
        # 存入 metrics 供 Callback (如 Checkpoint) 使用
        metric_key = "train_loss" if self.state == "TRAIN" else "val_loss"
        self.metrics[metric_key] = avg_loss
