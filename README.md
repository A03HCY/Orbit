# Orbit

Orbit is a flexible, plugin-based PyTorch training engine designed to simplify the training loop while providing powerful components for modern deep learning models, including LLMs.

It features a modular design with a rich set of plugins, advanced model building blocks (like MoE, RoPE, GQA), comprehensive LoRA/DoRA support, and cutting-edge optimizers.

## Features

### 🚀 Core Engine
- **Plugin System**: Decoupled training logic using plugins for callbacks, logging, and training strategies.
- **Simplified Loop**: Clean `train` and `eval` interfaces.
- **Flexible Updates**:
  - `auto_update()`: Automatically handles forward pass, loss calculation, backward pass, optimizer step, and zero grad.
  - `update(loss)`: Allows manual control over the update step if you need custom forward/loss logic.

### 🧩 Model Components (`orbit.model`)
Orbit provides a collection of high-performance, reusable layers:
- **Attention**: `MultiHeadAttention` with support for **GQA** (Grouped Query Attention), **RoPE** (Rotary Positional Embeddings), and FlashAttention.
- **LoRA & DoRA**: Full support for Low-Rank Adaptation and **Weight-Decomposed Low-Rank Adaptation (DoRA)** across `Linear`, `Conv2d`, `Conv1d`, and `Embedding` layers. Also supports **Gated LoRA**.
- **MoE**: Mixture of Experts block with `TopKGate` routing.
- **Gates**: A variety of gating mechanisms including `SigmoidGate`, `TanhGate`, `SoftmaxGate`, `GLUGate`, `ContextGate`, and `TopKGate`.
- **Others**: `FiLM` (Feature-wise Linear Modulation), `MLP` (with Gated support), `RotaryPositionalEmbedding`.

### 🛠️ Utilities & Kit (`orbit.utils`)

Orbit provides a comprehensive toolkit to speed up development:

#### 🔧 LoRA Utilities
Manual control over LoRA injection and management (alternative to the Plugin approach).
- **Injection**:
  - `inject_lora(model, r=8, ...)`: Manually inject LoRA/DoRA/Gated LoRA into specific layers.
  - `inject_lora_file(model, path)`: Automatically inject and load LoRA configuration/weights from a file.
- **Management**:
  - `merge_lora(model)` / `unmerge_lora(model)`: Merge weights for faster inference or unmerge to resume training.
  - `save_lora(model, path)` / `load_lora(model, path)`: Efficiently save/load only LoRA parameters.
  - `freeze_backbone_only(model)`: Helper to freeze the base model while keeping LoRA and specified heads trainable.
- **Diagnosis**:
  - `LoRADiagnoser`: Check for rank collapse and monitor gradient norms during training.

#### ❄️ Model Freezing
- `freeze_layers(model, targets=['encoder'])`: Freeze layers matching the target names (supports wildcards).
- `unfreeze_layers(model, targets)`: Unfreeze specific layers.
- `get_trainable_params(model)`: Get parameters for the optimizer.

#### 🎭 Masking
- `make_causal_mask`: Create causal masks for autoregressive models.
- `make_padding_mask`, `make_lookahead_mask`, `make_sliding_window_mask`.

#### 💾 Layer I/O
- `save_layer(model, layer_name, path)`: Save weights of a specific sub-module (e.g., just the backbone).
- `load_layer(model, layer_name, path)`: Load weights into a specific sub-module.
- `get_model_by_name(model, name)`: Access sub-modules using dot notation strings (e.g., "backbone.layer1").

#### 📝 SFT Helpers
- `build_sft`: Prepares data for Supervised Fine-Tuning (handles chat templates, tokenization, and label masking).
- `train_sft(engine)`: A specialized training step for SFT that handles the forward pass and loss calculation automatically.

#### ⚙️ Optimization (`orbit.optim`)
- **Muon**: MomentUm Orthogonalized by Newton-schulz optimizer.
- **SAM**: Sharpness-Aware Minimization wrapper.

#### 🌱 Initialization & Seeding
- `auto_initialize(model)`: Automatically initializes weights based on layer type (Linear, Conv, Embedding, etc.).
- `seed_everything(seed)`: Sets seeds for Python, NumPy, PyTorch, and CUDA for reproducibility.

#### 🖥️ CUDA
- `cuda_alloc(size)`: Optimizes PyTorch CUDA memory allocation configuration (e.g., `max_split_size_mb`).

### 🔌 Plugins (`orbit.plugin`)
- `EarlyStopping`: Stop training when a metric stops improving.
- `GradientAccumulation`: Simulate larger batch sizes.
- `Warmup`: Learning rate warmup.
- `Mentor`: Training assistant/logger.
- `MemoryEstimator`: Monitor CUDA memory usage.
- `LoRA`: Easy injection of LoRA layers via plugin.
- `Board`: TensorBoard integration.

## Installation

```bash
pip install .
```

**Requirements**:
- Python >= 3.8
- PyTorch >= 2.0.0 (Required for FlashAttention backend)

## Quick Start

### 1. Basic Training (CIFAR-10)

```python
import torch
import torch.nn as nn
from orbit.engine import Engine
from orbit.plugin import EarlyStopping, GradientAccumulation, Mentor
from orbit.utils import auto_initialize

# Define your model
model = MyConvNet()
auto_initialize(model)

# Setup Engine
trainer = Engine(
    model=model,
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    plugins=[
        Mentor(),
        EarlyStopping(monitor='val_acc', patience=3),
        GradientAccumulation(steps=2)
    ]
)

# Train
for _ in trainer.train(train_loader, num_epochs=10):
    trainer.auto_update() # Handles forward, backward, step, zero_grad
    
    # Handle Epoch End (e.g., Validation)
    if not trainer.is_epoch_end: continue
    
    for _ in trainer.eval(test_loader): 
        trainer.auto_update()
```

### 2. LLM SFT with LoRA/DoRA

Orbit makes it easy to fine-tune LLMs using LoRA or DoRA.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from orbit.engine import Engine
from orbit.plugin import LoRA, GradientAccumulation
from orbit.utils import train_sft, seed_everything

seed_everything(42)

# Load Model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

# Setup Engine with LoRA Plugin
trainer = Engine(
    model=model,
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
    plugins=[
        # Inject DoRA into MLP layers
        LoRA(target_names=['mlp'], dora=True, r=16, alpha=32),
        GradientAccumulation(steps=8)
    ]
)

# Train Loop
# Assuming `dataloader` yields SFT batches (input_ids, attention_mask, labels)
for _ in trainer.train(dataloader, num_epochs=3):
    # train_sft handles the forward pass and loss calculation for CausalLM
    train_sft(trainer) 
```

### 3. Chat Interface

Interact with your trained model in the terminal:

```python
from orbit.kit import ChatInterface

chat = ChatInterface(model_id="path/to/model", device="cuda")
chat.interact()
```

## License

MIT License
