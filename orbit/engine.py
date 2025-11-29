import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
import torch.nn as nn
from typing import Any, List, Optional, Union, Dict, Tuple

try: from torch.utils.tensorboard import SummaryWriter
except: pass

from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.console import Console

from orbit.callback import Callback, Forward
from orbit.plugin.checkpoint import Checkpoint
from orbit.plugin.board import Board
from orbit.plugin.display_model import ModelSummary

class Engine:
    '''训练循环控制器，负责协调模型训练、验证及回调事件。

    Engine 封装了 PyTorch 的训练循环，提供了插件机制（Callback），
    支持自动混合精度训练（AMP）、梯度裁剪、梯度累积、Checkpoint 保存、
    TensorBoard 可视化等功能。
    '''

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
        device_ids: Optional[List[int]] = None,
        use_amp: bool = False,
        grad_clip_norm: float = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        plugins: List[Callback] = None,
        forward_step: Optional[Forward] = None,
        checkpoint_dir: str = None,
        console: Console = None,
    ):
        '''初始化 Engine 实例。

        Args:
            model (nn.Module): 要训练的 PyTorch 模型。
            optimizer (torch.optim.Optimizer, optional): 优化器。如果为 None，则需要在其他地方（如插件）手动处理或稍后赋值。
            criterion (nn.Module, optional): 损失函数。如果为 None，则假设模型输出包含 loss 或自定义 loss 计算。
            device (Optional[str], optional): 运行设备 ('cpu', 'cuda', 'cuda:0' 等)。如果为 None，则自动检测。
            device_ids (Optional[List[int]], optional): GPU 设备 ID 列表。如果提供且长度 > 1，将启用 DataParallel。
            use_amp (bool, optional): 是否启用自动混合精度 (Automatic Mixed Precision) 训练。默认为 False。
            grad_clip_norm (float, optional): 梯度裁剪的范数阈值。如果为 None，则不进行梯度裁剪。
            scheduler (Optional[torch.optim.lr_scheduler._LRScheduler], optional): 学习率调度器。
            plugins (List[Callback], optional): 初始化时要挂载的回调插件列表。
            forward_step (Optional[Forward], optional): 自定义前向传播和 Loss 计算逻辑的实现。
            checkpoint_dir (str, optional): 快速设置 Checkpoint 保存目录的快捷参数。
            console (Console, optional): 用于输出日志的 Rich Console 实例。如果为 None，则创建一个新的。
        '''
        # --- 基础组件 ---
        self.device_ids = device_ids

        if self.device_ids and len(self.device_ids) > 0:
            self.device = torch.device(f"cuda:{self.device_ids[0]}")
        elif device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 移动模型到主设备
        model = model.to(self.device)

        # 多显卡处理 (DataParallel)
        if self.device_ids and len(self.device_ids) > 1:
            self.model = nn.DataParallel(model, device_ids=self.device_ids)
        else:
            self.model = model

        self.model_name = self.unwrap_model().__class__.__name__
        self.optimizer = optimizer
        self.criterion = criterion
        
        # --- 训练配置 ---
        self.use_amp = use_amp
        self.grad_clip_norm = grad_clip_norm
        self.scheduler = scheduler
        self.forward_step = forward_step
        self.scaler = torch.amp.GradScaler('cuda', enabled=use_amp) 

        # --- 交互与回调 ---
        self.console = console if console else Console()
        self.out_logs = self._OutLogs(self)
        self.writer: Optional[SummaryWriter] = None
        self.plugins = [
            ModelSummary(model),
        ]
        self.attach(plugins)
        
        if checkpoint_dir:
            self.attach(Checkpoint(name=self.model_name, path=checkpoint_dir))

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
    
    def unwrap_model(self) -> nn.Module:
        '''获取原始模型对象 (去除 DataParallel/DistributedDataParallel 包装)。'''
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self.model.module
        return self.model

    def is_in_warmup(self) -> bool:
        '''检查当前是否处于 Warmup 阶段。

        通过检查已挂载插件中是否存在 Warmup 插件，并判断当前全局步数是否在 Warmup 范围内。

        Returns:
            bool: 如果处于 Warmup 阶段返回 True，否则返回 False。
        '''
        for p in self.plugins:
            if p.__class__.__name__ == 'Warmup' and hasattr(p, 'total_warmup_steps'):
                if self.global_step <= p.total_warmup_steps:
                    return True
        return False

    def init_board(self, log_dir: str = 'runs') -> 'Engine':
        '''初始化 TensorBoard 可视化插件 (Board)。

        Args:
            log_dir (str, optional): TensorBoard 日志保存目录。默认为 'runs'。

        Returns:
            Engine: 返回 Engine 实例自身以支持链式调用。
        '''
        board = Board(name=self.model_name, log_dir=log_dir)
        self.attach(board, init=True)
        return self

    def set_checkpoint(self, dir: str, name: Optional[str] = None, **kwargs) -> 'Engine':
        '''配置 Checkpoint 插件。

        如果已存在 Checkpoint 插件，将被新配置替换。

        Args:
            dir (str): 模型保存目录。
            name (str, optional): 模型名称前缀。如果为 None，则使用 model_name。
            **kwargs: 传递给 Checkpoint 构造函数的其他参数 (如 monitor, save_top_k, mode 等)。

        Returns:
            Engine: 返回 Engine 实例自身以支持链式调用。
        '''
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
        '''统一日志打印方法，支持插件前缀。

        Args:
            *args: 要打印的内容。
            plugin (str, optional): 插件名称。如果提供，将以固定宽度和特定颜色打印前缀，
                用于区分不同来源的日志。
            **kwargs: 传递给 console.print 的其他参数。
        '''
        if plugin:
            # 宽度 15, 右对齐, 青色加粗
            prefix = f"[bold cyan]{plugin:>15}[/] │"
            self.console.print(prefix, *args, **kwargs)
        else:
            self.console.print(*args, **kwargs)
    
    def attach(self, plugin: Union[Callback, List[Callback]] = None, init: bool = False):
        '''挂载一个或多个插件到 Engine。

        Args:
            plugin (Union[Callback, List[Callback]], optional): 要挂载的插件或插件列表。
            init (bool, optional): 是否立即调用插件的 on_init 方法。
                通常在 Engine 初始化之后动态添加插件时设置为 True。默认为 False。

        Raises:
            ValueError: 如果传入的对象不是 Callback 实例。
        '''
        if not plugin: return
        if isinstance(plugin, Callback):
            plugin = [plugin]
        for p in plugin:
            if not isinstance(p, Callback):
                raise ValueError(f"Plugin {p} is not a Callback!")
            if p in self.plugins: continue
            if init: p.on_init(self)
            self.plugins.append(p)

    def _fire_event(self, event_name: str):
        '''触发所有已挂载插件的对应事件方法 (内部方法)。

        按插件挂载顺序依次调用。

        Args:
            event_name (str): 要触发的事件名称 (如 'on_epoch_start')。
        '''
        for cb in self.plugins:
            method = getattr(cb, event_name, None)
            if method:
                # [修改] 移除 try-except pass。
                # 我们需要看到 Callback 里的错误，否则调试是地狱。
                # 如果一定要防御性编程，可以使用 console.print_exception()
                method(self) 

    def _process_batch_data(self, batch_data: Any):
        '''处理 Batch 数据并将其移动到指定设备 (内部方法)。

        支持 Tensor, List[Tensor], Dict[str, Tensor] 等常见格式。
        自动解析并设置 self.data 和 self.target。

        Args:
            batch_data (Any): DataLoader 产生的一个 Batch 数据。
        '''
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

    def run(
        self,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        num_epochs: int = 10,
        start_epoch: Optional[int] = None,
        with_eval: bool = True
    ):
        '''启动训练循环。

        Args:
            train_loader (Any): 训练数据加载器 (通常是 torch.utils.data.DataLoader)。
            val_loader (Optional[Any], optional): 验证数据加载器。
            num_epochs (int, optional): 总训练轮数。默认为 10。
            start_epoch (Optional[int], optional): 起始 Epoch 索引。
                如果为 None，则从 0 开始。用于断点续训。
            with_eval (bool, optional): 是否在每个 Epoch 结束后执行验证。默认为 True。
        '''
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        if start_epoch is not None:
            self.start_epoch = start_epoch

        self._fire_event("on_train_start")
        try:
            for epoch in range(self.start_epoch, self.num_epochs):
                self.epoch = epoch
                self.metrics = {}
                
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

                # 打印 Epoch 总结
                lr_str = ""
                if self.optimizer:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    if current_lr < 1e-6:
                        lr_str = f" | LR: {current_lr:.2e}"
                    else:
                        lr_str = f" | LR: {current_lr:.6f}"
                    
                    if self.is_in_warmup():
                        lr_str += " [Warmup]"

                msg = f"Epoch {self.epoch+1}/{self.num_epochs}"
                if "train_loss" in self.metrics:
                    msg += f" | Train Loss: {self.metrics['train_loss']:.4f}"
                if "val_loss" in self.metrics:
                    msg += f" | Val Loss: {self.metrics['val_loss']:.4f}"
                msg += lr_str
                
                self.print(msg, plugin='Engine')
                
                if self.stop_training:
                    if self.epoch < self.num_epochs - 1:
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

    def _run_one_epoch(self, loader: Any, prefix: str = "Train", color: str = "blue"):
        '''执行单个 Epoch 的循环 (内部方法)。

        负责迭代 DataLoader，执行前向传播、反向传播（仅训练）、梯度更新及进度显示。

        Args:
            loader (Any): 数据加载器。
            prefix (str, optional): 进度条前缀显示文本。默认为 "Train"。
            color (str, optional): 进度条前缀颜色。默认为 "blue"。
        '''
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
                    if self.forward_step:
                        self.loss = self.forward_step.forward(self, self.data, self.target)
                    else:
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
                lr_str = ""
                if self.optimizer:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    if current_lr < 1e-6:
                        lr_str = f" LR: {current_lr:.2e}"
                    else:
                        lr_str = f" LR: {current_lr:.6f}"
                    
                    if self.is_in_warmup():
                        lr_str += " [Warmup]"

                logs = f"Loss: {loss_val:.4f}{lr_str} [Ep {self.epoch+1}/{self.num_epochs}]"
                progress.update(task, advance=1, description=logs)
                
                self._fire_event("on_batch_end")
        
        # 计算 epoch 平均 loss
        avg_loss = epoch_loss_sum / num_batches if num_batches > 0 else 0.0
        
        # 存入 metrics 供 Callback (如 Checkpoint) 使用
        metric_key = "train_loss" if self.state == "TRAIN" else "val_loss"
        self.metrics[metric_key] = avg_loss
