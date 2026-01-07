from orbit.utils import seed_everything, train_sft, cuda_alloc

seed_everything(42)
cuda_alloc(64)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = './model/qwen3-4b'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16, 
    device_map='auto'
)

from torch.utils.data import DataLoader
from torch.optim import Adam

optimizer = Adam(model.parameters(), lr=1e-4)

from sft import get_self_cognition_dataset

dataset = get_self_cognition_dataset(tokenizer, model_role='assistant')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

from orbit.engine import Engine
from orbit.plugin import LoRA, GradientAccumulation

trainer = Engine(
    model=model,
    optimizer=optimizer,
    plugins=[
        LoRA(
            target_names=[f'layers.{i}.mlp' for i in range(18, 26)],
            dora=True
        ),
        GradientAccumulation(steps=8)
    ]
)

for _ in trainer.train(dataloader, num_epochs=10):
    train_sft(trainer)
