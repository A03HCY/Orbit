import os
import torch
import inspect

from typing import Optional, List, TYPE_CHECKING
from orbit.callback import Event
from orbit.plugin.checkpoint import Checkpoint
from orbit.utils.lora import inject_lora, freeze_backbone_only

if TYPE_CHECKING: from orbit.engine import Engine

class LoRA(Checkpoint):
    '''LoRA 插件：集成 LoRA 注入、冻结、轻量化保存与加载功能。

    该插件继承自 Checkpoint，在训练开始时自动将模型转换为 LoRA 模型，
    重建优化器以适应新参数，并提供仅保存训练参数的轻量化 Checkpoint 功能。
    '''
    def __init__(
        self,
        name: str = "lora_model",
        path: str = "checkpoints",
        # LoRA 参数
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_names: Optional[List[str]] = None,
        exclude_names: Optional[List[str]] = None,
        unlock_head_keywords: Optional[List[str]] = None,
        gate: bool = False,
        dora: bool = False,
        # Checkpoint 参数
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_top_k: int = 1,
        save_last: bool = True,
        every_n_train_steps: Optional[int] = None,
        verbose: bool = True
    ):
        '''初始化 LoRA 插件。

        Args:
            name (str): 模型名称前缀。
            path (str): Checkpoint 保存目录。
            r (int): LoRA 秩。
            lora_alpha (int): LoRA 缩放系数。
            lora_dropout (float): LoRA Dropout。
            target_names (list, optional): 仅注入包含这些名称的层。
            exclude_names (list, optional): 排除包含这些名称的层。
            unlock_head_keywords (list, optional): 除了 LoRA 层外，还需要解冻的层关键字（如分类头）。
            gate (bool): 是否使用 Gated LoRA。
            dora (bool): 是否使用 DoRA。
            monitor (str): 监控指标。
            mode (str): 监控指标模式 ('min'/'max')。
            save_top_k (int): 保存最佳模型数量。
            save_last (bool): 是否保存最后的模型。
            every_n_train_steps (int, optional): 每 N 步保存一次。
            verbose (bool): 是否打印日志。
        '''
        # 初始化 Checkpoint，强制 save_weights_only=False 以保留训练状态，
        # 但我们会在 _save 中自定义过滤逻辑。
        super().__init__(
            name=name,
            path=path,
            save_weights_only=False,
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_last=save_last,
            every_n_train_steps=every_n_train_steps,
            verbose=verbose
        )
        
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_names = target_names
        self.exclude_names = exclude_names
        self.unlock_head_keywords = unlock_head_keywords
        self.gate = gate
        self.dora = dora
        
        self.injected = False

    def on_init(self, event: Event):
        engine = event.engine
        
        # 0. 冲突检测：检查是否存在其他 Checkpoint 插件
        other_checkpoints = [p for p in engine.plugins if isinstance(p, Checkpoint) and p is not self]
        if other_checkpoints:
            engine.print("[yellow]Warning: Multiple Checkpoint plugins detected. Since 'LoRA' inherits from 'Checkpoint', using both may cause conflicts (e.g. double saving). Suggest removing the standard 'Checkpoint' plugin.[/]", plugin='LoRA')

        model = engine.unwrap_model()
        
        # 1. 注入 LoRA 并冻结骨干
        if not self.injected:
            engine.print(f"[cyan]Injecting LoRA (r={self.r}, alpha={self.lora_alpha})...[/]", plugin='LoRA')
            inject_lora(
                model, 
                r=self.r, 
                lora_alpha=self.lora_alpha, 
                lora_dropout=self.lora_dropout,
                gate=self.gate,
                dora=self.dora,
                target_names=self.target_names,
                exclude_names=self.exclude_names
            )
            
            freeze_backbone_only(
                model, 
                unlock_head_keywords=self.unlock_head_keywords,
                verbose=self.verbose
            )
            
            model.to(engine.device)
            
            # 2. 重建 Optimizer
            # 由于参数集发生变化（新增了 LoRA 参数，冻结了大部分参数），旧的优化器无法使用。
            # 我们尝试使用旧优化器的类和 defaults 重新初始化。
            if engine.optimizer:
                old_opt = engine.optimizer
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                
                if not trainable_params:
                    engine.print("[red]Warning: No trainable parameters found after LoRA injection![/]", plugin='LoRA')
                else:
                    opt_cls = old_opt.__class__
                    defaults = old_opt.defaults
                    
                    # 过滤掉不在构造函数中的参数，防止某些库（如 transformers）在 defaults 中添加了额外元数据导致重建失败
                    sig = inspect.signature(opt_cls.__init__)
                    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                    if has_kwargs:
                        filtered_defaults = defaults
                    else:
                        filtered_defaults = {k: v for k, v in defaults.items() if k in sig.parameters}

                    engine.print(f"[cyan]Re-initializing Optimizer {opt_cls.__name__} for {len(trainable_params)} trainable groups...[/]", plugin='LoRA')
                    
                    # 创建新优化器
                    if opt_cls.__name__ == 'SAM':
                        base_opt_cls = old_opt.base_optimizer.__class__
                        new_opt = opt_cls(trainable_params, base_optimizer=base_opt_cls, **filtered_defaults)
                    elif opt_cls.__name__ == 'Muon':
                        muon_params = [p for p in trainable_params if p.ndim == 2]
                        adamw_params = [p for p in trainable_params if p.ndim != 2]
                        new_opt = opt_cls(muon_params=muon_params, adamw_params=adamw_params, **filtered_defaults)
                    else:
                        new_opt = opt_cls(trainable_params, **filtered_defaults)
                    
                    # 替换 Engine 中的优化器
                    engine.optimizer = new_opt
                    
                    # 处理 Scheduler 的关联
                    if engine.scheduler:
                        if hasattr(engine.scheduler, 'optimizer'):
                            engine.scheduler.optimizer = new_opt
                            engine.print("[yellow]Note: Scheduler optimizer reference updated.[/]", plugin='LoRA')
                        else:
                            engine.print("[yellow]Warning: Scheduler exists but cannot auto-update optimizer reference. Please verify scheduler behavior.[/]", plugin='LoRA')

            self.injected = True
            
        # 3. 执行 Checkpoint 的初始化 (创建目录, 尝试 Resume)
        super().on_init(event)

    def _save(self, engine: 'Engine', filename: str, verbose: bool = True, is_step: bool = False):
        """重写保存逻辑，仅保存可训练参数 (LoRA + Heads) 和训练状态。"""
        
        if self.monitor:
            engine.meta[self._meta_key] = {'best_k_models': self.best_k_models}

        raw_model = engine.unwrap_model()
        full_state_dict = raw_model.state_dict()
        lora_state_dict = {}
        
        # 筛选: 保存 requires_grad 的参数以及 LoRA/DoRA 相关的键
        for name, param in raw_model.named_parameters():
            if param.requires_grad:
                lora_state_dict[name] = full_state_dict[name]
        
        # 确保 buffers (如 BN running stats) 在解冻层中也被保存
        # 同时也保存所有名字中带 lora/dora 的 buffer (虽然通常它们没有 buffer)
        for key, value in full_state_dict.items():
            if 'lora_' in key or 'dora_' in key:
                if key not in lora_state_dict:
                    lora_state_dict[key] = value
        
        state = {
            'epoch': engine.epoch,
            'global_step': engine.global_step,
            'batch_idx': engine.batch_idx,
            'is_step': is_step,
            'model_state_dict': lora_state_dict,
            'optimizer_state_dict': engine.optimizer.state_dict() if engine.optimizer else None,
            'scheduler_state_dict': engine.scheduler.state_dict() if engine.scheduler else None,
            'scaler_state_dict': engine.scaler.state_dict() if engine.scaler else None,
            'meta': engine.meta,
            'orbit_lora_config': {
                'r': self.r,
                'alpha': self.lora_alpha,
                'target_names': self.target_names,
                'unlock_head_keywords': self.unlock_head_keywords
            }
        }
        
        file_path = os.path.join(self.path, filename)
        try:
            torch.save(state, file_path)
            if verbose:
                rel_path = os.path.relpath(file_path)
                file_size = os.path.getsize(file_path) / 1024 / 1024
                engine.print(f"Saved LoRA checkpoint: {rel_path} ({file_size:.2f} MB)", plugin='LoRA')
        except Exception as e:
            engine.print(f"[red]Failed to save checkpoint: {e}[/]", plugin='LoRA')

    def _load(self, engine: 'Engine', file_path: str):
        """重写加载逻辑，支持 strict=False 加载。"""
        engine.print(f"[cyan]Loading LoRA checkpoint from: {file_path}[/]", plugin='LoRA')
        try:
            checkpoint = torch.load(file_path, map_location=engine.device)
            raw_model = engine.unwrap_model()

            # 1. 加载模型参数
            model_sd = None
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model_sd = checkpoint['model_state_dict']
                
                # 配置检查
                saved_config = checkpoint.get('orbit_lora_config', {})
                if saved_config:
                    if saved_config.get('r') != self.r:
                        engine.print(f"[yellow]Warning: Loaded LoRA rank ({saved_config.get('r')}) != Current rank ({self.r})[/]", plugin='LoRA')
            else:
                # 兼容旧格式或纯权重文件
                model_sd = checkpoint if not isinstance(checkpoint, dict) else checkpoint

            if model_sd is not None:
                missing, unexpected = raw_model.load_state_dict(model_sd, strict=False)
                
                # 过滤掉骨干网络的缺失警告，只关注 LoRA 部分
                relevant_missing = [k for k in missing if 'lora_' in k or 'dora_' in k or any(h in k for h in (self.unlock_head_keywords or []))]
                if relevant_missing:
                    engine.print(f"[yellow]Warning: Missing relevant keys: {relevant_missing}[/]", plugin='LoRA')
                else:
                    engine.print("[green]LoRA weights loaded successfully.[/]", plugin='LoRA')

            # 2. 恢复训练状态 (仅当是完整的 Checkpoint 字典时)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                if engine.optimizer and 'optimizer_state_dict' in checkpoint:
                    try:
                        engine.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    except Exception as e:
                        engine.print(f"[yellow]Warning: Failed to load optimizer state: {e}.[/]", plugin='LoRA')

                if engine.scheduler and 'scheduler_state_dict' in checkpoint:
                    try:
                        engine.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    except: pass
                
                if engine.scaler and 'scaler_state_dict' in checkpoint:
                    engine.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                
                if 'meta' in checkpoint:
                    engine.meta.update(checkpoint['meta'])

                loaded_epoch = checkpoint.get('epoch', 0)
                loaded_batch_idx = checkpoint.get('batch_idx', -1)
                is_step = checkpoint.get('is_step', False)
                
                if is_step:
                    engine.start_epoch = loaded_epoch
                    engine.start_batch_idx = loaded_batch_idx
                    msg = f"Epoch {engine.start_epoch}, Batch {engine.start_batch_idx + 1}"
                else:
                    engine.start_epoch = loaded_epoch + 1
                    engine.start_batch_idx = -1
                    msg = f"Epoch {engine.start_epoch}"

                engine.global_step = checkpoint.get('global_step', 0)
                engine.print(f"[green]Resuming training from {msg}[/]", plugin='LoRA')
                
        except Exception as e:
            engine.print(f"[red]Failed to load checkpoint: {e}[/]", plugin='LoRA')
            import traceback
            traceback.print_exc()
