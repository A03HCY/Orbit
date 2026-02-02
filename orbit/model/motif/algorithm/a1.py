import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

from orbit.model import BaseBlock, register_model
from orbit.model.block import (
    MultiHeadAttention, AttentionOutput,
    MLP, RotaryPositionalEmbedding
)

@dataclass
class A1ModelOutput:
    logits: torch.Tensor
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

@register_model()
class A1Decoder(BaseBlock):
    ''' A1Decoder 模块。

    包含多头注意力机制和前馈神经网络。

    Args:
        model_dim (int, optional): 模型维度。默认为 512。
        num_heads (int, optional): 注意力头数。默认为 4。
        num_kv_heads (int, optional): 键/值头数。默认为 2。
        mlp_ratio (int, optional): MLP 隐藏层比率。默认为 2。
        dropout (float, optional): Dropout 概率。默认为 0.1。
    '''
    def __init__(
        self, 
        model_dim: int = 512, 
        num_heads: int = 4, 
        num_kv_heads: int = 2, 
        mlp_ratio: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.model_dim = model_dim

        self.attn_norm = nn.RMSNorm(model_dim)
        self.attn = MultiHeadAttention(
            hidden_size=model_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            use_qk_norm=True,
            use_gate=False,
            dropout=dropout
        )

        self.ffn_norm = nn.RMSNorm(model_dim)
        self.ffn = MLP(
            in_features=model_dim,
            hidden_features=int(mlp_ratio * model_dim),
            use_gate=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout, inplace=True)
    
    def forward(
        self, 
        x: torch.Tensor, 
        rope_emb: RotaryPositionalEmbedding, 
        mask: torch.Tensor = None,
        rotary_pos: int = 0,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> AttentionOutput:
        ''' 前向传播。

        Args:
            x (torch.Tensor): 输入张量。
            rope_emb (RotaryPositionalEmbedding): 旋转位置编码。
            mask (torch.Tensor, optional): 注意力掩码。默认为 None。
            rotary_pos (int, optional): 旋转位置编码起始位置。默认为 0。
            past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]], optional): 过去的键值对。默认为 None。
            use_cache (bool, optional): 是否使用缓存。默认为 False。

        Returns:
            AttentionOutput: 
                包含输出张量和过去的键值对。
        '''
        
        _x = self.attn_norm(x)
        attn: AttentionOutput = self.attn(
            hidden_states=_x,
            attention_mask=mask,
            rotary_emb=rope_emb,
            rotary_pos=rotary_pos,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        x = x + self.dropout(attn.output)

        _x = self.ffn_norm(x)
        ffn = self.ffn(_x)
        x = x + self.dropout(ffn)
        
        return AttentionOutput(
            output=x,
            past_key_value=attn.past_key_value
        )


@register_model()
class A1Model(BaseBlock):
    ''' A1Model。

    基于 Transformer Decoder 的语言模型。

    Args:
        vocab_size (int, optional): 词表大小。默认为 32000。
        model_dim (int, optional): 模型维度。默认为 512。
        num_layers (int, optional): 层数。默认为 8。
        num_heads (int, optional): 注意力头数。默认为 8。
        num_kv_heads (int, optional): 键/值头数。默认为 2。
        mlp_ratio (int, optional): MLP 隐藏层比率。默认为 2。
        dropout (float, optional): Dropout 概率。默认为 0.1。
        max_len (int, optional): 最大序列长度。默认为 2048。
        tie_weights (bool, optional): 是否共享 token_emb 与 proj_out 的权重。默认为 False。
    '''
    def __init__(
        self,
        vocab_size: int = 32000,
        model_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
        max_len: int = 2048,
        tie_weights: bool = False
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        
        self.token_emb = nn.Embedding(vocab_size, model_dim)
        self.rope_emb = RotaryPositionalEmbedding(
            model_dim=model_dim // num_heads,
            max_len=max_len,
            base=500000
        )
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            A1Decoder(
                model_dim=model_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.RMSNorm(model_dim)
        self.proj_out = nn.Linear(model_dim, vocab_size, bias=False)

        if tie_weights:
            self.proj_out.weight = self.token_emb.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                torch.nn.init.zeros_(module.weight[module.padding_idx])
        
    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor = None,
        start_pos: int = 0,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> A1ModelOutput:
        ''' 前向传播。

        Args:
            input_ids (torch.Tensor): 输入 Token ID 张量。Shape: [Batch, Seq_Len]。
            mask (torch.Tensor, optional): 注意力掩码。默认为 None。
            start_pos (int, optional): 起始位置，用于推理。默认为 0。
            past_key_values (Optional[List[Tuple[torch.Tensor, torch.Tensor]]], optional): 过去的键值对列表。默认为 None。
            use_cache (bool, optional): 是否使用缓存。默认为 False。

        Returns:
            A1ModelOutput: 模型输出。
        '''
        
        h = self.token_emb(input_ids)
        h = self.dropout(h)
        
        next_cache = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:
                def layer_wrapper(x, rope_emb, mask, rotary_pos, past_key_value, use_cache):
                    return layer(x, rope_emb, mask, rotary_pos, past_key_value, use_cache).output

                h = self.checkpoint(
                    layer_wrapper,
                    h,
                    self.rope_emb,
                    mask,
                    start_pos,
                    past_kv,
                    use_cache
                )
            else:
                output: AttentionOutput = layer(
                    x=h,
                    rope_emb=self.rope_emb,
                    mask=mask,
                    rotary_pos=start_pos,
                    past_key_value=past_kv,
                    use_cache=use_cache
                )
                h = output.output
                if use_cache:
                    next_cache.append(output.past_key_value)
                
        h = self.norm(h)
        logits = self.proj_out(h)
        
        return A1ModelOutput(
            logits=logits,
            past_key_values=next_cache
        )
