import torch
import torch
import numpy as np
import random
import os

def seed_everything(seed=42, strict=False):
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
    # 1. 设置 Python 原生 random
    random.seed(seed)
    
    # 2. 设置 Python 哈希种子 (影响字典/集合迭代顺序)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 3. 设置 Numpy
    np.random.seed(seed)
    
    # 4. 设置 PyTorch CPU/GPU
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 5. 设置 CuDNN 后端 (常规复现性设置)
    if torch.cuda.is_available():
        # 禁止寻找最优算法 (因为最优算法可能因硬件状态而变)
        torch.backends.cudnn.benchmark = False 
        # 强制使用确定性算法
        torch.backends.cudnn.deterministic = True 
    # 6. 严格模式 (Strict Mode)
    if strict:
        try:
            # 启用严格确定性算法
            # 注意：某些操作如果 PyTorch 没有对应的确定性实现，会直接通过 RuntimeError 报错
            torch.use_deterministic_algorithms(True)
            
            # 为了让 use_deterministic_algorithms 在 CUDA 上正常工作，
            # 必须设置 CUBLAS_WORKSPACE_CONFIG，否则会报 CuBLAS 错误。
            # :4096:8 是官方推荐的设置，虽然会增加少许显存开销
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