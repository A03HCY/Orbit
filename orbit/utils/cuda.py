import os

def cuda_alloc(size: int = 64):
    '''
    设置 PyTorch CUDA 内存分配配置

    Args:
        size (int): 最大分割大小（MB）
    '''
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:{size},expandable_segments:True'
