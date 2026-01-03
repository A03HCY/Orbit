import torch
import torch
import numpy as np
import random
import os

def seed_everything(seed=42, strict=False, warn_only=True):
    """
    设置所有随机种子以确保 PyTorch 实验的可复现性。
    
    Args:
        seed (int): 随机种子数值，默认为 42。
        strict (bool): 是否启用严格确定性模式。
                       如果为 True，将调用 torch.use_deterministic_algorithms(True)，
                       这可能会导致某些不支持确定性算法的操作报错，并且会降低训练速度。
    """
    import orbit
    orbit.seed_info = seed
    random.seed(seed)
    np.random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False 
        torch.backends.cudnn.deterministic = True 
    if strict:
        try:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            print(f"[Info] Strict deterministic mode enabled. (seed={seed})")
        except AttributeError:
            print("[Warning] torch.use_deterministic_algorithms is not available in your PyTorch version.")
    else:
        print(f"[Info] Random seed set as {seed}")

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_generator() -> torch.Generator:
    """创建随机数生成器"""
    import orbit
    seed = orbit.seed_info if hasattr(orbit, 'seed_info') else 42
    return torch.Generator().manual_seed(seed)