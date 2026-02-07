import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoTokenizer

from orbit.engine import Engine
from orbit.callback import Forward
from orbit.model.motif.algorithm.a1 import A1Model
from orbit.dataset import SFTDataset
from orbit.plugin import Checkpoint, Board, ModelSummary

# --- 1. 定义 SFT Forward 逻辑 ---
class SFTForward(Forward):
    '''自定义 SFT 前向传播，处理 Masked Loss'''
    def forward(self, engine, data, target):
        # data 是一个字典，由 SFTDataset 提供
        input_ids = data['input_ids']
        loss_mask = data['mask']
        
        # 模型前向传播
        outputs = engine.model(input_ids=input_ids)
        logits = outputs.logits
        
        # Shift (错位) 处理 Next Token Prediction
        # logits: [B, L, V] -> [B, L-1, V]
        shift_logits = logits[..., :-1, :].contiguous()
        # labels: [B, L] -> [B, L-1] (取 input_ids 作为标签)
        shift_labels = input_ids[..., 1:].contiguous()
        # mask:   [B, L] -> [B, L-1]
        shift_mask   = loss_mask[..., 1:].contiguous()
        
        # Flatten
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        flat_mask   = shift_mask.view(-1)
        
        # 计算 Loss (不 reduce)
        loss = F.cross_entropy(flat_logits, flat_labels, reduction='none')
        
        # 应用 Mask
        masked_loss = (loss * flat_mask).sum()
        denom = flat_mask.sum() + 1e-6
        
        return masked_loss / denom

# --- 2. Data Collator (Padding) ---
class SFTCollator:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        # batch 是一个 list，每个元素是 {'input_ids': Tensor, 'mask': Tensor}
        input_ids = [item['input_ids'] for item in batch]
        masks = [item['mask'] for item in batch]
        
        # Padding
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        masks_padded = torch.nn.utils.rnn.pad_sequence(
            masks, batch_first=True, padding_value=0 # mask padding 也是 0
        )
        
        return {
            'input_ids': input_ids_padded,
            'mask': masks_padded
        }

# --- 3. 配置与主函数 ---
def main():
    # 配置路径
    root_dir = "data" # 数据集根目录，请根据实际情况修改
    tokenizer_path = "gpt2" # 分词器路径，请根据实际情况修改
    output_dir = "checkpoints/sft_run"
    
    # 3.1 初始化组件
    print(f"Loading Tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    # 确保 pad_token_id 存在
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
        
    print("Initializing A1Model...")
    model = A1Model(
        vocab_size=len(tokenizer),
        model_dim=512, # 根据实际需求调整
        num_layers=8,
        num_heads=8
    )

    # 3.2 准备数据集 (手动组合)
    print("Scanning Datasets...")
    # en_sft_data 配置
    ds_en = SFTDataset(
        root_dir=root_dir,
        tokenizer=tokenizer,
        config={'sft': {'user': 'input', 'model': 'output'}},
        max_length=2048,
        in_memory=True # 如果数据量大可设为 False
    )
    
    # zh_sft_data 配置
    ds_zh = SFTDataset(
        root_dir=root_dir,
        tokenizer=tokenizer,
        config={'r1sft': {'user': 'instruction', 'model': 'output'}},
        max_length=2048,
        in_memory=True
    )
    
    # 混合数据集
    mixed_dataset = ConcatDataset([ds_en, ds_zh])
    
    print(f"Total samples: {len(mixed_dataset)}")
    
    if len(mixed_dataset) == 0:
        print("Warning: Dataset is empty. Please check 'root_dir' and data files.")
    
    # DataLoader
    loader = DataLoader(
        mixed_dataset, 
        batch_size=4, 
        shuffle=True, 
        collate_fn=SFTCollator(pad_token_id=tokenizer.pad_token_id),
        num_workers=0 # Windows 下多进程可能需要额外配置，先设为0
    )

    # 3.3 初始化 Engine
    engine = Engine(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-5),
        forward_step=SFTForward(), # <--- 注入自定义 Forward
        checkpoint_dir=output_dir,
        mixed_precision='no', # 可选 'fp16', 'bf16'
        grad_clip_norm=1.0
    )
    
    # 插件
    engine.attach([
        Board(name="SFT_Run"),
        ModelSummary(model)
    ])

    # 3.4 开始训练
    engine.run(train_loader=loader, num_epochs=3)

if __name__ == "__main__":
    main()
