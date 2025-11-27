import os
import torch
from typing import TYPE_CHECKING
from orbit.callback import Callback

if TYPE_CHECKING:
    from orbit.engine import Engine

class Checkpoint(Callback):
    def __init__(
        self, 
        name: str, 
        path: str, 
        save_weights_only: bool = False,
    ):
        super().__init__()
        self.name = name
        self.path = path
        self.save_weights_only = save_weights_only
        
    def on_init(self, engine: 'Engine'):
        """
        1. 创建文件夹
        """
        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)
        
        load_path = os.path.join(self.path, self.name + "_last.pt").replace("\\", "/")
        
        if os.path.exists(load_path):
            self._load(engine, load_path)
        else:
            engine.print(f"[yellow]Warning: Resume checkpoint '{load_path}' not found. Starting from scratch.[/]")

    def on_epoch_end(self, engine: 'Engine'):
        """每个 Epoch 保存一次 last"""
        self._save(engine, f"{self.name}_last.pt", verbose=False)
    
    def _save(self, engine: 'Engine', filename: str, verbose: bool = True):
        state = {
            'epoch': engine.epoch,
            'global_step': engine.global_step,
            'model_state_dict': engine.model.state_dict(),
            'optimizer_state_dict': engine.optimizer.state_dict() if engine.optimizer else None,
            'scheduler_state_dict': engine.scheduler.state_dict() if engine.scheduler else None,
            'scaler_state_dict': engine.scaler.state_dict() if engine.scaler else None,
            'meta': engine.meta, # 保存插件元数据
        }
        if self.save_weights_only:
            state = engine.model.state_dict()
        
        file_path = os.path.join(self.path, filename)
        try:
            torch.save(state, file_path)
            if verbose:
                engine.print(f"[green]Saved checkpoint: {file_path}[/]")
        except Exception as e:
            engine.print(f"[red]Failed to save checkpoint: {e}[/]")

    def _load(self, engine: 'Engine', file_path: str):
        """加载 Checkpoint 的核心逻辑"""
        engine.print(f"[cyan]Loading checkpoint from: {file_path}[/]")
        try:
            # 加载到设备 (GPU/CPU)
            checkpoint = torch.load(file_path, map_location=engine.device)
            
            # 1. 加载模型权重
            # 兼容两种格式：如果不包含 keys，说明整个文件就是 state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                engine.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                engine.model.load_state_dict(checkpoint)
                engine.print("[yellow]Loaded model weights only (legacy format).[/]")
                return # 如果是旧格式/纯权重，无法恢复 epoch 等，直接返回
            
            # 2. 如果不是纯权重模式，恢复训练状态 (Optimizer, Epoch等)
            if not self.save_weights_only:
                
                # 恢复 Optimizer
                if engine.optimizer and 'optimizer_state_dict' in checkpoint:
                    engine.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                # 恢复 Scheduler
                if engine.scheduler and 'scheduler_state_dict' in checkpoint:
                    engine.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                # 恢复 AMP Scaler
                if engine.scaler and 'scaler_state_dict' in checkpoint:
                    engine.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                
                # 恢复 Meta
                if 'meta' in checkpoint:
                    engine.meta.update(checkpoint['meta'])

                # 恢复 Epoch 和 Step
                # 注意：checkpoint 保存的是“完成时”的 epoch，所以开始应该 +1
                loaded_epoch = checkpoint.get('epoch', 0)
                engine.start_epoch = loaded_epoch + 1
                engine.global_step = checkpoint.get('global_step', 0)
                
                engine.print(f"[green]Successfully resumed training from Epoch {engine.start_epoch}, Global Step {engine.global_step}[/]")
                
        except Exception as e:
            engine.print(f"[red]Failed to load checkpoint: {e}[/]")
