import os
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, TYPE_CHECKING
from orbit.callback import Callback

if TYPE_CHECKING:
    from ..engine import Engine

class Board(Callback):
    def __init__(self, name: str, log_dir: str):
        super().__init__()
        self.log_dir = log_dir + '/' + name
        self.writer: Optional[SummaryWriter] = None

    def on_init(self, engine: 'Engine'):
        '''初始化 SummaryWriter'''
        # 如果 log_dir 不存在会自动创建
        self.writer = SummaryWriter(log_dir=self.log_dir)
        engine.writer = self.writer 
        engine.print(f'[cyan]TensorBoard initialized. Log dir: {self.log_dir}[/]')

    def on_batch_end(self, engine: 'Engine'):
        '''
        每个 Batch 结束时记录：
        1. Train Loss (Batch级)
        2. Learning Rate
        '''
        if engine.state == 'TRAIN':
            # 记录 Training Loss
            if engine.loss is not None:
                self.writer.add_scalar('Train/Batch_Loss', engine.loss.item(), engine.global_step)
            
            # 记录 Learning Rate (取第一个参数组)
            if engine.optimizer:
                current_lr = engine.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Train/LR', current_lr, engine.global_step)

    def on_epoch_end(self, engine: 'Engine'):
        '''
        每 Epoch 结束时记录：
        1. Epoch 平均 Loss (Train & Val)
        2. 其他 Metrics (如果在 engine.metrics 字典里有的话)
        '''
        
        # 我们根据 engine.metrics 里的内容自动记录
        # engine.metrics 应该长这样: {'train_loss': 0.5, 'val_loss': 0.4, 'acc': 0.9}
        for key, value in engine.metrics.items():
            # 自动把 'train_loss' 分组到 'Loss/train'
            if 'loss' in key.lower():
                tag = f'Loss/{key}'
            elif 'acc' in key.lower():
                tag = f'Accuracy/{key}'
            else:
                tag = f'Metrics/{key}'
            
            self.writer.add_scalar(tag, value, engine.epoch + 1)
            
        self.writer.flush()

    def on_train_end(self, engine: 'Engine'):
        '''训练结束关闭 Writer'''
        if self.writer:
            self.writer.close()