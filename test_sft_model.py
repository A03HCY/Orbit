from orbit.utils import seed_everything, train_sft

seed_everything(42)

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = './model/gemma3-1b'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto'
)

from torch.utils.data import DataLoader
from torch.optim import Adam

optimizer = Adam(model.parameters(), lr=2e-4)

from sft import get_self_cognition_dataset

dataset = get_self_cognition_dataset(tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

from orbit.engine import Engine
from orbit.plugin import LoRA, GradientAccumulation, Mentor

trainer = Engine(
    model=model,
    optimizer=optimizer,
    plugins=[
        LoRA(
            target_names=[f'layers.{i}.mlp' for i in range(18, 26)],
            dora=True
        ),
        GradientAccumulation(steps=32), Mentor()
    ]
)

for _ in trainer.train(dataloader, num_epochs=10):
    train_sft(trainer)
