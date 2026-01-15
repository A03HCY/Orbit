import torch
import torch.nn as nn
import torch.nn.functional as F

from orbit.model import BaseBlock, register_model
from orbit.model.block.mlp import MLP
from orbit.model.block.gate import TopKGate


@register_model()
class MoE(BaseBlock):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int = 4,
        top_k: int = 2,
        hidden_features: int = None,
        dropout: float = 0.1,
        use_gate: bool = False,
        use_mlp_router: bool = False
    ):
        super(MoE, self).__init__()

        hidden_features = hidden_features or in_features

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.dropout = dropout
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_gate = use_gate
        self.use_mlp_router = use_mlp_router

        self.router = TopKGate(
            in_features=in_features,
            out_features=num_experts,
            k=top_k,
            use_mlp=use_mlp_router,
            hidden_features=hidden_features,
            post_softmax=True
        )

        self.experts = nn.ModuleList([
            MLP(
                in_features=in_features,
                hidden_features=hidden_features,
                out_features=out_features,
                dropout=dropout,
                use_gate=use_gate
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。
        
        Args:
            x (torch.Tensor): 输入张量。Shape: [batch_size, seq_len, in_dim]
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: 
                - 输出张量。Shape: [batch_size, seq_len, out_features]
                - 辅助损失 (Auxiliary Loss)。标量。
        """
        batch_size, seq_len, dim = x.shape
        
        x_flat = x.view(-1, dim)
        
        gate_output = self.router(x_flat)
        
        routing_probs = F.softmax(gate_output.logits, dim=-1)
        selection_mask = torch.zeros_like(routing_probs).scatter_(1, gate_output.indices, 1.0)
        
        fraction = selection_mask.mean(dim=0)
        mean_probs = routing_probs.mean(dim=0)
        aux_loss = self.num_experts * (fraction * mean_probs).sum()
        
        final_output = torch.zeros(batch_size * seq_len, self.out_features, device=x.device, dtype=x.dtype)
        
        for i, expert in enumerate(self.experts):
            mask = (gate_output.indices == i)
            batch_idx, k_idx = torch.where(mask)
            
            if batch_idx.numel() == 0: continue
            
            inp = x_flat[batch_idx]
            expert_out = expert(inp)
            w = gate_output.values[batch_idx, k_idx].unsqueeze(-1)
            final_output.index_add_(0, batch_idx, expert_out * w)
            
        return final_output.view(batch_size, seq_len, self.out_features), aux_loss
