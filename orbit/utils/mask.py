import torch


def make_padding_mask(src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    '''
    创建填充掩码。

    Args:
        src (torch.Tensor): 源序列张量。形状为 [B, L_src]
        pad_idx (int, optional): 填充符号的索引。默认为 0。

    Returns:
        torch.Tensor: 填充掩码。形状为 [B, 1, 1, L_src]
        True 表示该位置不是填充，应该被关注。
    '''
    mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


def make_lookahead_mask(size: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    '''
    创建前瞻掩码（下三角矩阵）。

    Args:
        size (int): 序列长度。
        device (torch.device, optional): 设备。默认为 cpu。

    Returns:
        torch.Tensor: 前瞻掩码。形状为 [size, size]
        True 表示允许关注的位置（下三角部分）。
    '''
    mask = torch.tril(torch.ones((size, size), device=device)).bool()
    return mask


def make_causal_mask(tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    '''
    创建因果掩码（结合了填充掩码和前瞻掩码）。

    Args:
        tgt (torch.Tensor): 目标序列张量。形状为 [B, L_tgt]
        pad_idx (int, optional): 填充符号的索引。默认为 0。

    Returns:
        torch.Tensor: 因果掩码。形状为 [B, 1, L_tgt, L_tgt]
    '''
    pad_mask = make_padding_mask(tgt, pad_idx)
    seq_len = tgt.size(1)
    lookahead_mask = make_lookahead_mask(seq_len, device=tgt.device)

    # pad_mask: [B, 1, 1, L]
    # lookahead_mask: [L, L]
    # 广播后: [B, 1, L, L]
    mask = pad_mask & lookahead_mask
    return mask


def make_sliding_window_mask(
    tensor: torch.Tensor, window_size: int, pad_idx: int = 0, causal: bool = True
) -> torch.Tensor:
    '''
    创建滑动窗口掩码。

    Args:
        tensor (torch.Tensor): 输入序列张量。形状为 [B, L]
        window_size (int): 窗口大小（单侧）。
        pad_idx (int, optional): 填充符号的索引。默认为 0。
        causal (bool, optional): 是否为因果（单向）。默认为 True。
            如果为 True，位置 i 只能关注 [i - window_size, i]。
            如果为 False，位置 i 可以关注 [i - window_size, i + window_size]。

    Returns:
        torch.Tensor: 滑动窗口掩码。形状为 [B, 1, L, L]
    '''
    pad_mask = make_padding_mask(tensor, pad_idx)  # [B, 1, 1, L]
    seq_len = tensor.size(1)

    ones = torch.ones((seq_len, seq_len), device=tensor.device, dtype=torch.bool)

    if causal:
        # j <= i AND j >= i - window_size
        window_mask = torch.tril(ones, diagonal=0) & torch.triu(
            ones, diagonal=-window_size
        )
    else:
        # j <= i + window_size AND j >= i - window_size
        window_mask = torch.tril(ones, diagonal=window_size) & torch.triu(
            ones, diagonal=-window_size
        )

    mask = pad_mask & window_mask
    return mask
