import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import Any

from orbit.engine import Engine
from orbit.callback import Forward
from orbit.model.block.bio import PredictiveCodingBlock
from orbit.utils import auto_initialize
from orbit.plugin import ClassificationReport

# --- 1. Data Preparation ---
batch_size = 64

norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # MNIST mean and std
])

train_data = torchvision.datasets.MNIST(
    root='../data/mnist',
    train=True,
    download=True,
    transform=norm
)
test_data = torchvision.datasets.MNIST(
    root='../data/mnist',
    train=False,
    download=True,
    transform=norm
)

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)
test_loader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False
)

# --- 2. Model Definition ---
class PCNet(nn.Module):
    def __init__(self, in_features=784, num_classes=10, hidden_dims=[512, 256]):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.total_features = in_features + num_classes
        
        # Use the new PredictiveCodingBlock
        self.pc_block = PredictiveCodingBlock(
            in_features=self.total_features,
            hidden_dims=hidden_dims,
            num_iter=30,
            lr_state=0.1,
            lr_weight=0.001,
            auto_update=True,
            output_activations=[nn.ReLU(), nn.ReLU()]
        )

    def forward(self, x, mask=None):
        # x: (B, 794)
        # mask: (B, 794)
        return self.pc_block(x, mask)

    def predict(self, x, mask=None):
        return self.pc_block.predict(x, mask)

# --- 3. Custom Forward Logic ---
class PCForward(Forward):
    def forward(self, engine: Engine, data: Any, target: Any) -> torch.Tensor:
        model = engine.unwrap_model()
        batch_size = data.size(0)
        img_flat = data.view(batch_size, -1)
        
        if engine.state == "TRAIN":
            # Training: [Image | Label]
            label_onehot = F.one_hot(target, num_classes=10).float().to(data.device)
            x_in = torch.cat([img_flat, label_onehot], dim=1)
            
            # Forward (updates weights internally)
            pc_output = engine.model(x_in)
            state1 = pc_output.output
            
            # Calculate reconstruction error (Layer 1)
            error = model.pc_block.get_prediction_error(x_in, state1)
            loss = torch.mean(error ** 2)
            return loss
            
        else:
            # Inference: [Image | Zeros] + Mask
            label_zeros = torch.zeros(batch_size, 10, device=data.device)
            x_in = torch.cat([img_flat, label_zeros], dim=1)
            
            mask_img = torch.ones_like(img_flat)
            mask_label = torch.zeros_like(label_zeros)
            mask = torch.cat([mask_img, mask_label], dim=1)
            
            # Predict
            pred_x = model.predict(x_in, mask)
            pred_logits = pred_x[:, -10:]
            
            engine.output = pred_logits
            engine.target = target
            loss = F.cross_entropy(pred_logits, target)
            return loss

# --- 4. Training Setup ---
model = PCNet()
auto_initialize(model)

trainer = Engine(
    model=model,
    optimizer=None, 
    forward_step=PCForward(),
    plugins=[
        ClassificationReport(num_classes=10, top_k=1)
    ]
)

print("Starting Hierarchical Predictive Coding training on MNIST...")

# --- 5. Training Loop ---
for _ in trainer.train(train_loader, num_epochs=5):
    trainer.auto_update()

    if not trainer.is_epoch_end: continue
    for _ in trainer.eval(test_loader):
        trainer.auto_update()

print("Training finished.")
