import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
import torch.nn as nn
from typing import Any, List, Optional, Union, Dict, Tuple

try: from torch.utils.tensorboard import SummaryWriter
except: pass

from rich.progress  import Progress, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.console   import Console
from accelerate     import Accelerator

from orbit.callback import Callback, Forward, Event
from orbit.plugin   import Checkpoint, Board, ModelSummary
from orbit.utils    import load_model


class Engine:
    '''训练循环控制器，负责协调模型训练、验证及回调事件。

    Engine 封装了 PyTorch 的训练循环，并深度集成了 Accelerate 库。
    它提供了开箱即用的分布式训练支持（DDP, FSDP, DeepSpeed 等）、
    自动混合精度（AMP）、梯度裁剪、梯度累积、Checkpoint 管理以及 TensorBoard 可视化等功能。
    '''

    class _OutLogs:
        def __init__(self, engine: 'Engine'):
            self.engine = engine
        def __enter__(self):
            # self.engine._print_edge(top=False)
            self.engine.console.print('\n')
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.engine.console.print('\n')
            # self.engine._print_edge(top=True)

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer = None,
        criterion: nn.Module = None,
        accelerator: Optional[Accelerator] = None,
        mixed_precision: str = 'no', # 'no', 'fp16', 'bf16'
        grad_clip_norm: float = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        plugins: List[Callback] = None,
        forward_step: Optional[Forward] = None,
        checkpoint_dir: str = None,
        console: Console = None,
        **accelerator_kwargs
    ):
        '''初始化 Engine 实例。

        Args:
            model (nn.Module): 要训练的 PyTorch 模型。
            optimizer (torch.optim.Optimizer, optional): 优化器。如果为 None，则需要在其他地方手动处理。
            criterion (nn.Module, optional): 损失函数。如果为 None，则假设模型输出包含 loss 或自定义 loss 计算。
            accelerator (Optional[Accelerator], optional): 预初始化的 Accelerator 实例。
                如果为 None，将使用 mixed_precision 和 accelerator_kwargs 创建一个新的实例。
            mixed_precision (str, optional): 混合精度模式。可选值为 'no', 'fp16', 'bf16'。默认为 'no'。
            grad_clip_norm (float, optional): 梯度裁剪的范数阈值。如果为 None，则不进行梯度裁剪。
            scheduler (Optional[torch.optim.lr_scheduler._LRScheduler], optional): 学习率调度器。
            plugins (List[Callback], optional): 初始化时要挂载的回调插件列表。
            forward_step (Optional[Forward], optional): 自定义前向传播和 Loss 计算逻辑的实现。
            checkpoint_dir (str, optional): 快速设置 Checkpoint 保存目录的快捷参数。
            console (Console, optional): 用于输出日志的 Rich Console 实例。如果为 None，则创建一个新的。
            **accelerator_kwargs: 传递给 Accelerator 构造函数的其他参数（如 cpu, split_batches 等）。
        '''
        # --- Accelerator 初始化 ---
        if accelerator:
            self.accelerator = accelerator
        else:
            self.accelerator = Accelerator(mixed_precision=mixed_precision, **accelerator_kwargs)
        
        # --- 基础组件 ---
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        
        # 准备模型、优化器和调度器
        # 注意：DataLoader 将在 run/train/eval 中准备
        if self.optimizer:
            if self.scheduler:
                self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.scheduler
                )
            else:
                self.model, self.optimizer = self.accelerator.prepare(
                    self.model, self.optimizer
                )
        else:
            self.model = self.accelerator.prepare(self.model)

        self.model_name = self.unwrap_model().__class__.__name__
        
        # --- 训练配置 ---
        self.grad_clip_norm = grad_clip_norm
        self.forward_step = forward_step

        # --- 交互与回调 ---
        self.prepare  = self.accelerator.prepare
        self.backward = self.accelerator.backward
        self.autocast = self.accelerator.autocast

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
        self.start_batch_idx = -1 # 恢复训练时的起始 Batch 索引 (跳过此索引及之前的)
        self.is_first_batch = False
        self.is_last_batch = False
        self.is_end_of_epoch = False
        self.is_epoch_end = False
        
        self.state = "IDLE"      # TRAIN / EVAL
        self.stop_training = False # 插件可以通过设置此标志为 True 来停止训练
        self.stop_source: Optional[str] = None
        self.stop_reason: Optional[str] = None
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

    @property
    def device(self):
        '''获取当前使用的设备。'''
        return self.accelerator.device

    @property
    def use_amp(self) -> bool:
        '''是否启用了混合精度训练。'''
        return self.accelerator.mixed_precision != 'no'

    @property
    def scaler(self):
        '''获取 GradScaler 对象（如果存在）。'''
        return self.accelerator.scaler

    def stop(self, source: str = "User", reason: str = "Unknown"):
        '''请求停止训练。

        Args:
            source (str): 停止请求的来源 (例如 "EarlyStopping", "User", "KeyboardInterrupt")。
            reason (str): 停止的具体原因。
        '''
        self.stop_training = True
        self.stop_source = source
        self.stop_reason = reason
    
    def unwrap_model(self) -> nn.Module:
        '''获取原始模型对象。
        
        使用 accelerator.unwrap_model 去除 DDP/FSDP/DeepSpeed 等分布式包装，
        返回原始的 nn.Module 实例。

        Returns:
            nn.Module: 原始模型对象。
        '''
        return self.accelerator.unwrap_model(self.model)

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

        注意：此操作仅在主进程 (Main Process) 中执行，以避免多进程写入冲突。

        Args:
            log_dir (str, optional): TensorBoard 日志保存目录。默认为 'runs'。

        Returns:
            Engine: 返回 Engine 实例自身以支持链式调用。
        '''
        # 仅在主进程初始化 Board
        if self.accelerator.is_main_process:
            board = Board(name=self.model_name, log_dir=log_dir)
            self.attach(board, init=True)
        return self

    def set_checkpoint(self, dir: str, name: Optional[str] = None, **kwargs) -> 'Engine':
        '''配置 Checkpoint 插件。

        如果已存在 Checkpoint 插件，将被新配置替换。
        注意：此操作仅在主进程 (Main Process) 中执行。

        Args:
            dir (str): 模型保存目录。
            name (str, optional): 模型名称前缀。如果为 None，则使用 model_name。
            **kwargs: 传递给 Checkpoint 构造函数的其他参数 (如 monitor, save_top_k, mode 等)。

        Returns:
            Engine: 返回 Engine 实例自身以支持链式调用。
        '''
        # 仅在主进程配置 Checkpoint
        if not self.accelerator.is_main_process:
            return self

        if name is None:
            name = self.model_name
            
        self.plugins = [p for p in self.plugins if not isinstance(p, Checkpoint)]
        
        ckpt = Checkpoint(name=name, path=dir, **kwargs)
        ckpt.on_init(Event(engine=self, name="on_init"))
        self.attach(ckpt)
        return self
    
    def _print_edge(self, top=True):
        if not self.accelerator.is_main_process: return
        char = '┬' if top else '┴'
        self.console.print(' ' + '─' * 15 + char + '─' * 35)
    
    def print(self, *args, plugin: Optional[str] = None, **kwargs):
        '''统一日志打印方法，支持插件前缀。

        注意：此方法仅在主进程 (Main Process) 中输出日志，其他进程的调用将被忽略。

        Args:
            *args: 要打印的内容。
            plugin (str, optional): 插件名称。如果提供，将以固定宽度和特定颜色打印前缀，
                用于区分不同来源的日志。
            **kwargs: 传递给 console.print 的其他参数。
        '''
        if not self.accelerator.is_main_process: return
        
        if plugin:
            # 宽度 15, 右对齐, 青色加粗
            prefix = f"[[bold cyan]{plugin:>15}[/]] "
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
            if init: p.on_init(Event(engine=self, name="on_init"))
            self.plugins.append(p)

    def _fire_event(self, event_name: str, **kwargs):
        '''触发所有已挂载插件的对应事件方法 (内部方法)。

        按插件挂载顺序依次调用。

        Args:
            event_name (str): 要触发的事件名称 (如 'on_epoch_start')。
            **kwargs: 传递给 Event 构造函数的其他参数 (如 source, reason)。
        '''
        event = Event(engine=self, name=event_name, **kwargs)
        for cb in self.plugins:
            method = getattr(cb, event_name, None)
            if method:
                method(event) 

    def _process_batch_data(self, batch_data: Any):
        '''处理 Batch 数据并将其移动到指定设备 (内部方法)。

        支持 Tensor, List[Tensor], Dict[str, Tensor] 等常见格式。
        自动解析并设置 self.data 和 self.target。
        
        注意：如果 DataLoader 已经被 accelerator.prepare 处理，数据通常已经在正确设备上。
        此方法包含额外的检查以确保数据位于 self.device 上。

        Args:
            batch_data (Any): DataLoader 产生的一个 Batch 数据。
        '''
        def to_device(data):
            if isinstance(data, torch.Tensor):
                return data.to(self.device)
            elif isinstance(data, (list, tuple)):
                return [to_device(x) for x in data]
            elif isinstance(data, dict):
                return {k: to_device(v) for k, v in data.items()}
            return data

        # 如果数据不在当前设备上，移动它
        # 注意：accelerator.prepare 后的 DataLoader 通常已经处理了设备放置
        
        if isinstance(batch_data, (list, tuple)):
            # 简单检查第一个元素
            if len(batch_data) > 0 and isinstance(batch_data[0], torch.Tensor) and batch_data[0].device != self.device:
                batch_data = to_device(batch_data)
                
            if len(batch_data) == 2:
                self.data, self.target = batch_data
            elif len(batch_data) == 1:
                self.data = batch_data[0]
                self.target = None
            else:
                self.data = batch_data[:-1]
                self.target = batch_data[-1]
        elif isinstance(batch_data, dict):
            # 简单检查第一个值
            first_val = next(iter(batch_data.values()), None)
            if isinstance(first_val, torch.Tensor) and first_val.device != self.device:
                batch_data = to_device(batch_data)
                
            self.data = batch_data
            self.target = None 
        else:
            if isinstance(batch_data, torch.Tensor) and batch_data.device != self.device:
                batch_data = batch_data.to(self.device)
            self.data = batch_data
            self.target = None

    def update(self, loss: torch.Tensor):
        '''执行反向传播及参数更新。

        使用 accelerator.backward 处理反向传播，支持自动混合精度和梯度累积。
        同时支持 SAM (Sharpness-Aware Minimization) 优化器的两步更新逻辑。

        Args:
            loss (torch.Tensor): 当前 Step 的 Loss。
        '''
        if self.is_epoch_end: return
        
        original_loss = loss
        self.loss = loss

        if self.optimizer is None:
            self.global_step += 1
            return

        # 梯度累积处理
        backward_loss = loss
        if self.accumulation_steps > 1:
            backward_loss = loss / self.accumulation_steps
        
        self.accelerator.backward(backward_loss)

        if (self.batch_idx + 1) % self.accumulation_steps == 0 or self.is_last_batch:
            
            # 检测是否为 SAM 优化器 (Duck Typing)
            is_sam = hasattr(self.optimizer, 'first_step') and hasattr(self.optimizer, 'second_step')

            if is_sam:
                # --- SAM Optimizer Logic ---
                # SAM 需要两次 backward，accelerate 支持多次 backward
                
                if self.grad_clip_norm:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                
                # First Step: Compute e_w
                self.optimizer.first_step(zero_grad=True)

                # Second Forward-Backward
                self._forward_pass()
                self.accelerator.backward(self.loss)
                
                if self.grad_clip_norm:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                
                # Second Step: Update weights
                self.optimizer.second_step(zero_grad=True)
                
                self.loss = original_loss

            else:
                # --- Standard Optimizer Logic ---
                if self.grad_clip_norm:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            self.global_step += 1

    def _forward_pass(self) -> torch.Tensor:
        '''执行前向传播并计算 Loss (内部方法)。
        
        使用 accelerator.autocast() 上下文管理器自动处理混合精度。
        '''
        # 使用 accelerator.autocast() 处理混合精度
        with self.accelerator.autocast():
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
                    self.loss = self.criterion(self.output, self.target)
                else:
                    self.loss = torch.tensor(0.0, device=self.device)
            
            return self.loss

    def auto_update(self) -> torch.Tensor:
        '''自动执行前向传播、Loss 计算、反向传播及参数更新。
        
        如果在评估模式 (EVAL) 下调用，仅执行前向传播和 Loss 计算。

        Returns:
            torch.Tensor: 当前 Step 的 Loss (未缩放)。
        '''
        loss = self._forward_pass()
        
        # 仅在训练模式下执行更新
        if self.state == "TRAIN":
            self.update(loss)
            
        return loss

    def run(
        self,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        num_epochs: int = 10,
        start_epoch: Optional[int] = None,
        with_eval: bool = True
    ):
        '''启动训练循环。

        此方法会自动使用 accelerator.prepare 包装数据加载器以支持分布式训练。

        Args:
            train_loader (Any): 训练数据加载器 (通常是 torch.utils.data.DataLoader)。
            val_loader (Optional[Any], optional): 验证数据加载器。
            num_epochs (int, optional): 总训练轮数。默认为 10。
            start_epoch (Optional[int], optional): 起始 Epoch 索引。
                如果为 None，则从 0 开始。用于断点续训。
            with_eval (bool, optional): 是否在每个 Epoch 结束后执行验证。默认为 True。
        '''
        # 准备 DataLoaders
        if val_loader:
            train_loader, val_loader = self.accelerator.prepare(train_loader, val_loader)
        else:
            train_loader = self.accelerator.prepare(train_loader)
            
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

                if self.stop_training:
                    if self.epoch < self.num_epochs - 1:
                        source = self.stop_source if self.stop_source else "Plugin"
                        reason = self.stop_reason if self.stop_reason else "Unknown"
                        self.print(f"[yellow]Training stopped by {source}: {reason}[/]", plugin='Engine')
                        self._fire_event("on_requested_stop", source=source, reason=reason)
                    break

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

                msg = f"[dark_magenta]Epoch {self.epoch+1}/{self.num_epochs}"
                if "train_loss" in self.metrics:
                    msg += f" | Train Loss: {self.metrics['train_loss']:.4f}"
                if "val_loss" in self.metrics:
                    msg += f" | Val Loss: {self.metrics['val_loss']:.4f}"
                msg += lr_str
                
                self.print(msg, plugin='Engine')
                    
        except KeyboardInterrupt:
            self.print("[red][bold]Training interrupted by user.", plugin='Engine')
            self.stop(source="User", reason="KeyboardInterrupt")
            self._fire_event("on_requested_stop", source="User", reason="KeyboardInterrupt")
        except Exception as e:
            self.exception = e
            self.console.print_exception()
            self._fire_event("on_exception")
        finally:
            self._fire_event("on_train_end")

    def _train_epoch_iterator(self, loader: Any, total_steps: Optional[int] = None, prefix: str = "Train", color: str = "blue"):
        '''生成器：执行单个 Epoch 的训练循环。
        
        注意：进度条仅在主进程中显示。
        '''
        self.model.train()
        self.is_epoch_end = False
        torch.cuda.empty_cache()
        
        try:
            real_len = len(loader)
        except:
            real_len = None
            
        num_batches = total_steps if total_steps is not None else real_len
        
        # 仅主进程显示进度条
        disable_progress = not self.accelerator.is_main_process
        
        with Progress(
            TextColumn(f"[{color}]{prefix}"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=True,
            disable=disable_progress
        ) as progress:
            
            task = progress.add_task(f"[Ep {self.epoch+1}/{self.num_epochs}]", total=num_batches)
            
            for batch_idx, batch_data in enumerate(loader):
                if self.epoch == self.start_epoch and batch_idx <= self.start_batch_idx:
                    if not disable_progress:
                        progress.update(task, advance=1, description=f"[dim]Skipping batch {batch_idx}...[/]")
                    continue

                self.batch_idx = batch_idx
                self.is_first_batch = (batch_idx == 0)
                
                if real_len is not None:
                    self.is_last_batch = (batch_idx == real_len - 1)
                elif num_batches is not None:
                    self.is_last_batch = (batch_idx == num_batches - 1)
                else:
                    self.is_last_batch = False
                
                self._process_batch_data(batch_data)
                self._fire_event("on_batch_start")

                yield self

                loss_val = self.loss.item() if self.loss is not None else 0.0
                
                if not disable_progress:
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
                
                if self.stop_training: break
            
        if not self.stop_training:
            self.is_epoch_end = True
            yield self
            self.is_epoch_end = False

    def _eval_epoch_iterator(self, loader: Any, total_steps: Optional[int] = None, prefix: str = "Eval ", color: str = "yellow"):
        '''生成器：执行单个 Epoch 的验证/测试循环。
        
        注意：进度条仅在主进程中显示。
        '''
        self.model.eval()
        
        try:
            real_len = len(loader)
        except:
            real_len = None
            
        num_batches = total_steps if total_steps is not None else real_len
        disable_progress = not self.accelerator.is_main_process
        
        with Progress(
            TextColumn(f"[{color}]{prefix}"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=True,
            disable=disable_progress
        ) as progress:
            lr_str = ""
            if self.optimizer:
                current_lr = self.optimizer.param_groups[0]['lr']
                if current_lr < 1e-6:
                    lr_str = f" LR: {current_lr:.2e}"
                else:
                    lr_str = f" LR: {current_lr:.6f}"
                
                if self.is_in_warmup():
                    lr_str += " [Warmup]"
            
            task = progress.add_task(f"{lr_str} [Ep {self.epoch+1}/{self.num_epochs}]", total=num_batches)
            
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(loader):
                    self.batch_idx = batch_idx
                    self.is_first_batch = (batch_idx == 0)
                    
                    if real_len is not None:
                        self.is_last_batch = (batch_idx == real_len - 1)
                    elif num_batches is not None:
                        self.is_last_batch = (batch_idx == num_batches - 1)
                    else:
                        self.is_last_batch = False
                    
                    self._process_batch_data(batch_data)
                    self._fire_event("on_batch_start")

                    yield self
                    
                    loss_val = self.loss.item() if self.loss is not None else 0.0
                    if not disable_progress:
                        logs = f"Loss: {loss_val:.4f}{lr_str} [Ep {self.epoch+1}/{self.num_epochs}]"
                        progress.update(task, advance=1, description=logs)
                    self._fire_event("on_batch_end")

    def train(
        self,
        train_loader: Any,
        num_epochs: int = 10,
        start_epoch: Optional[int] = None,
        total_steps: Optional[int] = None
    ):
        '''生成器：启动训练循环，允许用户自定义 Step 逻辑。

        此方法会自动使用 accelerator.prepare 包装数据加载器。

        Args:
            train_loader (Any): 训练数据加载器。
            num_epochs (int, optional): 总训练轮数。
            start_epoch (int, optional): 起始 Epoch。
            total_steps (int, optional): 手动指定进度条的总步数 (用于特殊 Loader)。
        '''
        train_loader = self.accelerator.prepare(train_loader)
        self.train_loader = train_loader
        self.num_epochs = num_epochs
        if start_epoch is not None:
            self.start_epoch = start_epoch

        self._fire_event("on_train_start")
        try:
            for epoch in range(self.start_epoch, self.num_epochs):
                self.epoch = epoch
                self.metrics = {}
                self.state = "TRAIN"
                
                self._fire_event("on_epoch_start")
                
                epoch_loss_sum = 0.0
                count = 0
                
                for _ in self._train_epoch_iterator(self.train_loader, total_steps=total_steps):
                    yield self
                    if self.loss is not None and not self.is_epoch_end:
                        epoch_loss_sum += self.loss.item()
                        count += 1
                
                if self.stop_training:
                    break

                if self.scheduler:
                    self.scheduler.step()

                self._fire_event("on_epoch_end")
                
                avg_loss = epoch_loss_sum / count if count > 0 else 0.0
                self.metrics['train_loss'] = avg_loss
                
                lr_str = ""
                if self.optimizer:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    if current_lr < 1e-6:
                        lr_str = f" | LR: {current_lr:.2e}"
                    else:
                        lr_str = f" | LR: {current_lr:.6f}"
                    if self.is_in_warmup():
                        lr_str += " [Warmup]"

                msg = f"[dark_magenta]Epoch {self.epoch+1}/{self.num_epochs} | Train Loss: {avg_loss:.4f}{lr_str}"
                self.print(msg, plugin='Engine')

        except KeyboardInterrupt:
            self.print("[red][bold]Training interrupted by user.", plugin='Engine')
            self.stop(source="User", reason="KeyboardInterrupt")
            self._fire_event("on_requested_stop", source="User", reason="KeyboardInterrupt")
        except Exception as e:
            self.exception = e
            self.console.print_exception()
            self._fire_event("on_exception")
        finally:
            self._fire_event("on_train_end")

    def eval(
        self,
        val_loader: Any,
        total_steps: Optional[int] = None,
        description: str = "Eval "
    ):
        '''生成器：启动评估循环。

        此方法会自动使用 accelerator.prepare 包装数据加载器。

        Args:
            val_loader (Any): 验证数据加载器。
            total_steps (int, optional): 手动指定进度条的总步数。
            description (str, optional): 进度条描述。
        '''
        val_loader = self.accelerator.prepare(val_loader)
        self.val_loader = val_loader
        self.state = "EVAL"
        self._fire_event("on_eval_start")
        
        try:
            epoch_loss_sum = 0.0
            count = 0
            
            for _ in self._eval_epoch_iterator(self.val_loader, total_steps=total_steps, prefix=description):
                yield self
                if self.loss is not None:
                    epoch_loss_sum += self.loss.item()
                    count += 1
            
            avg_loss = epoch_loss_sum / count if count > 0 else 0.0
            self.metrics['val_loss'] = avg_loss
            
        except KeyboardInterrupt:
            self.print("[red][bold]Eval interrupted by user.", plugin='Engine')
            self.stop(source="User", reason="KeyboardInterrupt")
            self._fire_event("on_requested_stop", source="User", reason="KeyboardInterrupt")
        except Exception as e:
            self.exception = e
            self.console.print_exception()
            self._fire_event("on_exception")
        finally:
            self._fire_event("on_eval_end")

    def _run_one_epoch(self, loader: Any, prefix: str = "Train", color: str = "blue"):
        '''执行单个 Epoch 的循环 (内部方法)。'''
        is_train = (self.state == "TRAIN")
        
        if is_train:
            epoch_loss_sum = 0.0
            count = 0
            for _ in self._train_epoch_iterator(loader, prefix=prefix, color=color):
                if self.is_epoch_end: continue
                self.auto_update()
                epoch_loss_sum += self.loss.item()
                count += 1
            
            avg_loss = epoch_loss_sum / count if count > 0 else 0.0
            self.metrics['train_loss'] = avg_loss
        
        else:
            epoch_loss_sum = 0.0
            count = 0
            for _ in self._eval_epoch_iterator(loader, prefix=prefix, color=color):
                self._forward_pass()
                epoch_loss_sum += self.loss.item()
                count += 1
            
            avg_loss = epoch_loss_sum / count if count > 0 else 0.0
            self.metrics['val_loss'] = avg_loss
    
    def load_checkpoint(self, path: str) -> 'Engine':
        plugin: Checkpoint = [p for p in self.plugins if not isinstance(p, Checkpoint)][0]
        plugin._load(self, path)
        return self

    def load(self, path: str, strict: bool = True) -> 'Engine':
        '''从文件加载模型权重。

        Args:
            path (str): 权重文件路径。
            strict (bool): 是否严格匹配键值。默认为 True。
        
        Returns:
            Engine: 返回 Engine 实例自身以支持链式调用。
        '''
        load_model(self.unwrap_model(), path, strict=strict, map_location=self.device)
        return self
