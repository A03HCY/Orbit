import os
import torch
from typing import TYPE_CHECKING, List, Tuple
from orbit.callback import Callback

if TYPE_CHECKING:
    from orbit.engine import Engine

class Checkpoint(Callback):
    def __init__(
        self, 
        name: str, 
        path: str, 
        save_weights_only: bool = False,
        monitor: str = 'val_loss', # 默认监控 val_loss
        mode: str = 'min',         # 默认 loss 越小越好
        save_top_k: int = 1,
        save_last: bool = True,
        verbose: bool = True
    ):
        """
        Args:
            name (str): 模型名称前缀。
            path (str): 保存目录。
            save_weights_only (bool): 是否只保存模型权重 (不保存 optimizer 等状态)。
            monitor (str): 监控指标 (例如 'val_loss', 'val_acc')。默认 'val_loss'。
            mode (str): 'min' (越小越好) 或 'max' (越大越好)。
            save_top_k (int): 保存最好的 K 个模型。设为 0 则禁用 Top-K 保存。
            save_last (bool): 是否总是保存 '{name}_last.pt'。
            verbose (bool): 是否打印保存信息。
        """
        super().__init__()
        self.name = name
        self.path = path
        self.save_weights_only = save_weights_only
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.verbose = verbose
        
        # 维护 Top-K 模型列表: [(score, filename), ...]
        self.best_k_models: List[Tuple[float, str]] = []
        
        # 内部状态 Key
        self._meta_key = 'checkpoint_callback'

    def on_init(self, engine: 'Engine'):
        """
        1. 创建文件夹
        2. 尝试恢复 Checkpoint 状态 (best_k_models)
        3. 尝试加载 last checkpoint
        """
        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)
        
        # 尝试恢复 best_k_models 状态
        if self._meta_key in engine.meta:
            self.best_k_models = engine.meta[self._meta_key].get('best_k_models', [])

        load_path = os.path.join(self.path, self.name + "_last.pt").replace("\\", "/")
        
        if os.path.exists(load_path):
            self._load(engine, load_path)
        else:
            engine.print(f"[yellow]Warning: Resume checkpoint '{load_path}' not found. Starting from scratch.[/]", plugin='Checkpointing')

    def on_epoch_end(self, engine: 'Engine'):
        """
        每个 Epoch 结束时：
        1. 保存 last
        2. 如果设置了 monitor，保存 top_k
        """
        # 1. Save Last
        if self.save_last:
            self._save(engine, f"{self.name}_last.pt", verbose=False) # last 不需要每次都啰嗦
        
        # 2. Save Top K
        if self.monitor and self.save_top_k > 0:
            current_score = engine.metrics.get(self.monitor)
            
            if current_score is None:
                if self.verbose:
                    engine.print(f"[yellow]Metric '{self.monitor}' not found in metrics. Skipping Top-K save.[/]", plugin='Checkpointing')
                return

            self._check_and_save_top_k(engine, current_score)

    def _check_and_save_top_k(self, engine: 'Engine', current_score: float):
        """检查并保存 Top-K 模型"""
        filename = f"{self.name}_ep{engine.epoch+1}_{self.monitor}_{current_score:.4f}.pt"
        
        # 逻辑简化：总是先加入，然后排序，如果超过 K 个，删除最差的
        self.best_k_models.append((current_score, filename))
        
        # 排序
        reverse = (self.mode == 'max')
        self.best_k_models.sort(key=lambda x: x[0], reverse=reverse)
        
        # 如果列表过长，处理溢出
        if len(self.best_k_models) > self.save_top_k:
            worst_model = self.best_k_models.pop() # 移除最后一个（最差的）
            worst_score, worst_filename = worst_model
            
            # 如果刚才加入的就是最差的，说明没进 Top K，不需要保存
            if worst_filename == filename:
                return 
            
            # 否则，保存新的，删除旧的最差的
            self._save(engine, filename, verbose=self.verbose)
            self._remove(engine, worst_filename)
        else:
            # 列表没满，直接保存
            self._save(engine, filename, verbose=self.verbose)
            
        # 更新 Meta 状态
        engine.meta[self._meta_key] = {'best_k_models': self.best_k_models}

    def _save(self, engine: 'Engine', filename: str, verbose: bool = True):
        # 确保 meta 数据是最新的
        if self.monitor:
            engine.meta[self._meta_key] = {'best_k_models': self.best_k_models}

        # 获取原始模型 (去除 DataParallel 包装) 以保证 Checkpoint 通用性
        raw_model = engine.unwrap_model()

        state = {
            'epoch': engine.epoch,
            'global_step': engine.global_step,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': engine.optimizer.state_dict() if engine.optimizer else None,
            'scheduler_state_dict': engine.scheduler.state_dict() if engine.scheduler else None,
            'scaler_state_dict': engine.scaler.state_dict() if engine.scaler else None,
            'meta': engine.meta,
        }
        if self.save_weights_only:
            state = raw_model.state_dict()
        
        file_path = os.path.join(self.path, filename)
        try:
            torch.save(state, file_path)
            if verbose:
                # 使用相对路径显示，更简洁
                rel_path = os.path.relpath(file_path)
                engine.print(f"Saved checkpoint: {rel_path}[/]", plugin='Checkpointing')
        except Exception as e:
            engine.print(f"[red]Failed to save checkpoint: {e}[/]", plugin='Checkpointing')

    def _remove(self, engine: 'Engine', filename: str):
        """删除旧的 Checkpoint 文件"""
        file_path = os.path.join(self.path, filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                if self.verbose:
                    engine.print(f"[dim]Removed old checkpoint: {filename}[/]", plugin='Checkpointing')
            except OSError as e:
                engine.print(f"[red]Failed to remove checkpoint {filename}: {e}[/]", plugin='Checkpointing')

    def _load(self, engine: 'Engine', file_path: str):
        """加载 Checkpoint 的核心逻辑"""
        engine.print(f"[cyan]Loading checkpoint from: {file_path}[/]", plugin='Checkpointing')
        try:
            # 加载到设备
            checkpoint = torch.load(file_path, map_location=engine.device)
            
            # 获取原始模型以进行加载
            raw_model = engine.unwrap_model()

            # 1. 加载模型权重
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                raw_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                raw_model.load_state_dict(checkpoint)
                engine.print("[yellow]Loaded model weights only (legacy format).[/]", plugin='Checkpointing')
                return 
            
            # 2. 恢复训练状态
            if not self.save_weights_only:
                if engine.optimizer and 'optimizer_state_dict' in checkpoint:
                    engine.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                if engine.scheduler and 'scheduler_state_dict' in checkpoint:
                    engine.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                if engine.scaler and 'scaler_state_dict' in checkpoint:
                    engine.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                
                if 'meta' in checkpoint:
                    engine.meta.update(checkpoint['meta'])

                loaded_epoch = checkpoint.get('epoch', 0)
                engine.start_epoch = loaded_epoch + 1
                engine.global_step = checkpoint.get('global_step', 0)
                
                engine.print(f"[green]Successfully resumed training from Epoch {engine.start_epoch}, Global Step {engine.global_step}[/]", plugin='Checkpointing')
                
        except Exception as e:
            engine.print(f"[red]Failed to load checkpoint: {e}[/]", plugin='Checkpointing')
