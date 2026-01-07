import torch
import torch.nn as nn
import inspect
from typing import Optional, Any, List, Union


class AutoRegressiveWrapper:
    '''
    一个将普通 torch.nn.Module 包装为兼容 transformers generate 接口的包装类

    Attributes:
        model (nn.Module): 原始模型实例
        device (torch.device): 模型所在的设备
        accepts_attention_mask (bool): 模型是否接受 attention_mask 参数
        accepts_mask (bool): 模型是否接受 mask 参数
    '''

    def __init__(self, model: nn.Module):
        '''
        初始化包装器

        Args:
            model (nn.Module): 只有 forward 方法的自定义模型
        '''
        self.model = model
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device('cpu')

        sig = inspect.signature(model.forward)
        params = sig.parameters
        self.accepts_attention_mask = 'attention_mask' in params
        self.accepts_mask = 'mask' in params
        
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        streamer: Optional[Any] = None,
        **kwargs
    ) -> torch.Tensor:
        '''
        自回归生成循环，支持多种采样策略并兼容 TextIteratorStreamer

        Args:
            input_ids (torch.Tensor): 输入的 token ID 序列 [batch, seq_len]
            max_new_tokens (int): 最大新生成的 token 数量
            temperature (float): 采样温度
            top_k (int): Top-k 采样的 k 值
            top_p (float): Top-p (Nucleus) 采样的 p 值
            repetition_penalty (float): 重复惩罚系数
            do_sample (bool): 是否使用采样
            eos_token_id (Optional[Union[int, List[int]]]): 终止 token ID
            streamer (Optional[Any]): transformers 库的 streamer 实例
            **kwargs: 忽略其他 transformers 相关的参数

        Returns:
            torch.Tensor: 包含生成内容的完整序列
        '''
        curr_input_ids = input_ids.to(self.device)
        batch_size = curr_input_ids.shape[0]
        
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=self.device)

        for _ in range(max_new_tokens):
            model_inputs = {'input_ids': curr_input_ids}

            if self.accepts_attention_mask:
                model_inputs['attention_mask'] = torch.ones_like(curr_input_ids)
            elif self.accepts_mask:
                model_inputs['mask'] = torch.ones_like(curr_input_ids)

            outputs = self.model(**model_inputs)

            if isinstance(outputs, (tuple, list)):
                logits = outputs[0]
            else:
                logits = outputs

            next_token_logits = logits[:, -1, :]

            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in set(curr_input_ids[i].tolist()):
                        if next_token_logits[i, previous_token] < 0:
                            next_token_logits[i, previous_token] *= repetition_penalty
                        else:
                            next_token_logits[i, previous_token] /= repetition_penalty

            if do_sample:
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            curr_input_ids = torch.cat([curr_input_ids, next_token], dim=-1)

            if streamer is not None:
                if unfinished_sequences[0] == 1:
                    streamer.put(next_token.cpu())

            if eos_token_id is not None:
                for token_id in eos_token_id:
                    unfinished_sequences = unfinished_sequences.mul(
                        next_token.tile(1, 1).ne(token_id).all(dim=-1).long()
                    )

            if unfinished_sequences.max() == 0:
                break

        if streamer is not None:
            streamer.end()

        return curr_input_ids

    def __getattr__(self, name: str) -> Any:
        '''
        将未定义的属性访问转发给原始模型

        Args:
            name (str): 属性名称

        Returns:
            Any: 原始模型的属性
        '''
        return getattr(self.model, name)
