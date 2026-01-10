import torch
from typing import Tuple

def split_to_patches(image: torch.Tensor, patch_size: Tuple[int, int]) -> torch.Tensor:
    '''将图像张量分割成多个子图像（补丁）。

    此函数接收形状为 [..., channels, width, height] 的图像张量，
    并将其划分为形状为 [channels, patch_width, patch_height] 的补丁。
    结果张量的形状为 [..., num_patches, channels, patch_width, patch_height]。

    Args:
        image (torch.Tensor): 输入图像张量，形状为 [..., channels, w, h]。
            最后两个维度被视为空间维度（宽度，高度）。
        patch_size (Tuple[int, int]): 表示补丁大小的元组 (a, b)，其中 'a' 对应于
            宽度维度，'b' 对应于高度维度。

    Returns:
        torch.Tensor: 形状为 [..., num, channels, a, b] 的补丁张量，
            其中 num 是补丁的总数 ((w // a) * (h // b))。

    Raises:
        ValueError: 如果输入图像的空间维度不能被补丁大小整除。
    '''
    if image.ndim < 3:
        raise ValueError(f'Input image must have at least 3 dimensions, got {image.ndim}')

    w, h = image.shape[-2], image.shape[-1]
    a, b = patch_size

    if w % a != 0 or h % b != 0:
        raise ValueError(
            f'Image dimensions ({w}, {h}) must be divisible by patch size ({a}, {b})'
        )

    reshaped = image.view(*image.shape[:-2], w // a, a, h // b, b)
    permuted = reshaped.permute(*range(reshaped.ndim - 5), reshaped.ndim - 4, reshaped.ndim - 2, reshaped.ndim - 5, reshaped.ndim - 3, reshaped.ndim - 1)
    num_patches = (w // a) * (h // b)

    return permuted.reshape(*image.shape[:-3], num_patches, image.shape[-3], a, b)
