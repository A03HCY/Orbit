import torch
import torch.nn.functional as F
from typing import Tuple
from dataclasses import dataclass

@dataclass
class PatchOutput:
    output: torch.Tensor
    mask: torch.Tensor
    patch_size: Tuple[int, int]
    num_patches: Tuple[int, int]

def split_to_patches(image: torch.Tensor, patch_size: Tuple[int, int]) -> PatchOutput:
    '''将图像张量分割成多个子图像，支持自动填充。

    此函数接收形状为 [..., channels, width, height] 的图像张量，
    并将其划分为形状为 [channels, patch_width, patch_height] 的补丁。
    如果图像尺寸不能被 patch_size 整除，则在右侧和底部进行零填充。
    结果张量的形状为 [..., num_patches_total, channels, patch_width, patch_height]。

    Args:
        image (torch.Tensor): 输入图像张量，形状为 [..., channels, w, h]。
            最后两个维度被视为空间维度（宽度，高度）。
        patch_size (Tuple[int, int]): 表示补丁大小的元组 (a, b)，其中 'a' 对应于
            宽度维度，'b' 对应于高度维度。

    Returns:
        PatchOutput: 包含以下字段的数据类：
            - output (torch.Tensor): 形状为 [..., num_w * num_h, channels, a, b] 的补丁张量。
            - mask (torch.Tensor): 形状为 [..., num_w * num_h, 1, a, b] 的掩码张量，
              有效区域为 1，填充区域为 0。
            - patch_size (Tuple[int, int]): 输入的补丁大小 (a, b)。
            - num_patches (Tuple[int, int]): 宽度和高度方向的补丁数量 (num_w, num_h)。

    Raises:
        ValueError: 如果输入图像维度少于 3。
    '''
    if image.ndim < 3:
        raise ValueError(f'Input image must have at least 3 dimensions, got {image.ndim}')

    w, h = image.shape[-2], image.shape[-1]
    a, b = patch_size

    pad_w = (a - w % a) % a
    pad_h = (b - h % b) % b

    image_padded = F.pad(image, (0, pad_h, 0, pad_w))

    w_padded = w + pad_w
    h_padded = h + pad_h

    num_w = w_padded // a
    num_h = h_padded // b

    mask = torch.ones((*image.shape[:-3], 1, w, h), dtype=image.dtype, device=image.device)
    mask_padded = F.pad(mask, (0, pad_h, 0, pad_w), value=0)

    def split(x, nw, nh):
        # x shape: [..., C, W, H]
        reshaped = x.view(*x.shape[:-2], nw, a, nh, b)
        # permute to [..., nw, nh, C, a, b]
        permuted = reshaped.permute(
            *range(reshaped.ndim - 5),
            reshaped.ndim - 4, # nw
            reshaped.ndim - 2, # nh
            reshaped.ndim - 5, # C
            reshaped.ndim - 3, # a
            reshaped.ndim - 1  # b
        )
        return permuted.reshape(*x.shape[:-3], nw * nh, x.shape[-3], a, b)

    output_patches = split(image_padded, num_w, num_h)
    mask_patches = split(mask_padded, num_w, num_h)

    return PatchOutput(
        output=output_patches,
        mask=mask_patches,
        patch_size=patch_size,
        num_patches=(num_w, num_h)
    )

def reconstruct_from_patches(patches: torch.Tensor, num_patches: Tuple[int, int], mask: torch.Tensor = None) -> torch.Tensor:
    '''从补丁重建图像。

    此函数是 split_to_patches 的逆操作。它将补丁张量重新组合成原始图像。
    如果提供了 mask，则会根据 mask 去除 padding。

    Args:
        patches (torch.Tensor): 形状为 [..., num_patches, channels, patch_width, patch_height] 的补丁张量。
        num_patches (Tuple[int, int]): 宽度和高度方向的补丁数量 (num_w, num_h)。
        mask (torch.Tensor, optional): 用于去除 padding 的掩码，形状与 patches 相同但通道数为 1。
            如果提供，将根据掩码裁剪重建后的图像以去除 padding。

    Returns:
        torch.Tensor: 重建后的图像张量，形状为 [..., channels, width, height]。
    '''
    nw, nh = num_patches
    if patches.shape[-4] != nw * nh:
         raise ValueError(f"Number of patches in tensor ({patches.shape[-4]}) does not match num_patches argument ({nw} * {nh} = {nw*nh})")

    a, b = patches.shape[-2], patches.shape[-1]
    
    def unsplit(x):
        # x: [..., nw*nh, C, a, b]
        reshaped = x.view(*x.shape[:-4], nw, nh, *x.shape[-3:])
        # permute to [..., C, nw, a, nh, b]
        permuted = reshaped.permute(
            *range(reshaped.ndim - 5),
            reshaped.ndim - 3, # C
            reshaped.ndim - 5, # nw
            reshaped.ndim - 2, # a
            reshaped.ndim - 4, # nh
            reshaped.ndim - 1  # b
        )
        return permuted.reshape(*x.shape[:-4], x.shape[-3], nw * a, nh * b)

    reconstructed = unsplit(patches)
    
    if mask is not None:
        reconstructed_mask = unsplit(mask)
        m = reconstructed_mask.view(-1, reconstructed_mask.shape[-2], reconstructed_mask.shape[-1])
        
        valid_h = (m[0, 0, :] > 0.5).sum().item()
        valid_w = (m[0, :, 0] > 0.5).sum().item()
        
        reconstructed = reconstructed[..., :int(valid_w), :int(valid_h)]

    return reconstructed
