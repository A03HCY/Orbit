import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn.attention import SDPBackend, sdpa_kernel

from orbit.model import BaseBlock
from orbit.model.block import RotaryPositionalEmbedding

def apply_attention(
    query_states: torch.Tensor, 
    key_states: torch.Tensor, 
    value_states: torch.Tensor, 
    attention_mask: torch.Tensor = None, 
    output_attentions: bool = False,
    is_causal: bool = None
) -> dict[str, torch.Tensor]:
    ''' 计算缩放点积注意力。

    Args:
        query_states (torch.Tensor): 查询状态张量。
        key_states (torch.Tensor): 键状态张量。
        value_states (torch.Tensor): 值状态张量。
        attention_mask (torch.Tensor, optional): 注意力掩码。默认为 None。
        output_attentions (bool, optional): 是否输出注意力权重。默认为 False。
        is_causal (bool, optional): 是否应用因果掩码。默认为 None。

    Returns:
        dict[str, torch.Tensor]: 包含注意力输出和可选权重的字典。
    '''
    
    if not output_attentions:
        if attention_mask is None and is_causal is None: is_causal = True
        
        try:
            with sdpa_kernel([
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION
            ]):
                output = F.scaled_dot_product_attention(
                    query_states, 
                    key_states, 
                    value_states, 
                    attn_mask=attention_mask if not is_causal else None, 
                    is_causal=is_causal,
                    dropout_p=0.0
                )
            return {'output': output, 'attention_weights': None}
        except Exception:
            print('Error at attn cal')
            pass

    d_k = query_states.size(-1)
    scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(d_k)

    if attention_mask is not None:
        scores = scores.masked_fill(attention_mask == 0, float('-inf'))

    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value_states)
    
    return {'output': output, 'attention_weights': attention_weights}


class MultiHeadAttention(BaseBlock):
    ''' 多头注意力机制模块。

    支持分组查询注意力 (GQA)、QK 归一化和门控机制。

    Args:
        hidden_size (int): 隐藏层大小。
        num_heads (int): 注意力头数。
        num_kv_heads (int, optional): 键/值头数，用于 GQA。如果为 None，则等于 num_heads。默认为 None。
        use_qk_norm (bool, optional): 是否对查询和键应用 RMSNorm。默认为 True。
        use_gate (bool, optional): 是否应用门控机制。默认为 False。
    '''

    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads=None,
        use_qk_norm=True,
        use_gate=False
    ):
        super(MultiHeadAttention, self).__init__()

        if num_kv_heads is None: num_kv_heads = num_heads

        assert hidden_size % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.hidden_size = hidden_size
        self.num_heads  = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_queries = num_heads // num_kv_heads
        self.head_dim  = hidden_size // num_heads
        self.use_qk_norm = use_qk_norm
        self.use_gate = use_gate

        if use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, hidden_size)

        self.q_proj = nn.Linear(hidden_size, hidden_size)

        kv_dim = self.num_kv_heads * self.head_dim
        self.k_proj = nn.Linear(hidden_size, kv_dim)
        self.v_proj = nn.Linear(hidden_size, kv_dim)

        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        output_attentions: bool = False,
        rotary_emb: RotaryPositionalEmbedding = None,
        rotary_pos: int = 0,
        past_key_value: tuple[torch.Tensor, torch.Tensor] = None,
        use_cache: bool = False
    ) -> dict[str, torch.Tensor]:
        ''' 执行多头注意力的前向传播。

        Args:
            hidden_states (torch.Tensor): 输入隐藏状态。
            kv_states (torch.Tensor, optional): 用于键/值的隐藏状态。如果为 None，则使用 hidden_states。默认为 None。
            attention_mask (torch.Tensor, optional): 注意力掩码。默认为 None。
            output_attentions (bool, optional): 是否输出注意力权重。默认为 False。
            rotary_emb (RotaryPositionalEmbedding, optional): 旋转位置编码模块。默认为 None。
            rotary_pos (int, optional): 旋转位置编码的起始位置。默认为 0。
            past_key_value (tuple[torch.Tensor, torch.Tensor], optional): 过去的键值对缓存。默认为 None。
            use_cache (bool, optional): 是否使用 KV 缓存。默认为 False。

        Returns:
            dict[str, torch.Tensor]: 包含输出、注意力权重和 KV 缓存的字典。
        '''
        
        if kv_states is None:
            kv_states = hidden_states

        batch_size, q_len, _ = hidden_states.shape
        kv_len_input = kv_states.shape[1] 

        if self.use_gate:
            G = self.g_proj(hidden_states)
        
        Q = self.q_proj(hidden_states)
        K = self.k_proj(kv_states)
        V = self.v_proj(kv_states)

        Q = Q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, kv_len_input, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, kv_len_input, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            Q = self.q_norm(Q)
            K = self.k_norm(K)
        
        if rotary_emb is not None:
            Q = rotary_emb(Q, start_pos=rotary_pos)
            K = rotary_emb(K, start_pos=rotary_pos)
        
        current_key_value = None
        if use_cache:
            if past_key_value is not None:
                past_k, past_v = past_key_value
                K = torch.cat((past_k, K), dim=2)
                V = torch.cat((past_v, V), dim=2)
            current_key_value = (K, V)

        kv_seq_len_total = K.shape[2]

        if self.num_kv_queries > 1:
            # [B, H_kv, 1, L, D] -> [B, H_kv, G, L, D]
            K = K[:, :, None, :, :].expand(batch_size, self.num_kv_heads, self.num_kv_queries, kv_seq_len_total, self.head_dim)
            V = V[:, :, None, :, :].expand(batch_size, self.num_kv_heads, self.num_kv_queries, kv_seq_len_total, self.head_dim)
            
            K = K.reshape(batch_size, self.num_heads, kv_seq_len_total, self.head_dim)
            V = V.reshape(batch_size, self.num_heads, kv_seq_len_total, self.head_dim)

        attn_output = apply_attention(Q, K, V, attention_mask, output_attentions)
        output = attn_output['output']
        attention_weights = attn_output['attention_weights']

        output = output.transpose(1, 2).contiguous().view(batch_size, q_len, self.hidden_size)

        output = self.o_proj(output)

        if self.use_gate:
            output = output * torch.sigmoid(G)

        return {
            'output': output,
            'attention_weights': attention_weights,
            'past_key_value': current_key_value
        }
