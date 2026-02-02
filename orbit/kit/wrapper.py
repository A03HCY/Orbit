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
        accepts_use_cache (bool): 模型是否接受 use_cache 参数
        accepts_past_key_values (bool): 模型是否接受 past_key_values 参数
        accepts_start_pos (bool): 模型是否接受 start_pos 参数
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
        self.accepts_use_cache = 'use_cache' in params
        self.accepts_past_key_values = 'past_key_values' in params
        self.accepts_start_pos = 'start_pos' in params
        
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
        input_ids = input_ids.to(self.device)
        curr_input_ids = input_ids
        batch_size = curr_input_ids.shape[0]
        
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=self.device)
        
        # 初始掩码处理
        mask = kwargs.get('attention_mask')
        if mask is None:
            mask = kwargs.get('mask')
        
        if mask is not None:
            mask = mask.to(self.device)
        
        past_key_values = None
        start_pos = 0
        use_cache = self.accepts_use_cache and self.accepts_past_key_values

        for i in range(max_new_tokens):
            if use_cache and past_key_values is not None:
                model_inputs = {'input_ids': curr_input_ids[:, -1:]}
            else:
                model_inputs = {'input_ids': curr_input_ids}

            # 动态更新掩码
            current_mask = None
            if mask is not None:
                if mask.shape[1] < curr_input_ids.shape[1]:
                    padding = torch.ones(
                        (batch_size, curr_input_ids.shape[1] - mask.shape[1]), 
                        dtype=mask.dtype, 
                        device=self.device
                    )
                    mask = torch.cat([mask, padding], dim=1)
                current_mask = mask
            elif not use_cache:
                # 如果没有提供掩码且不使用缓存（即全序列计算），
                # 通常让模型内部处理因果掩码（例如传入 None）
                current_mask = None
            else:
                # 在缓存模式下，如果一直没有掩码，可以保持为 None 以触发模型默认逻辑
                current_mask = None

            if self.accepts_attention_mask:
                model_inputs['attention_mask'] = current_mask
            elif self.accepts_mask:
                model_inputs['mask'] = current_mask
            
            if use_cache:
                model_inputs['use_cache'] = True
                model_inputs['past_key_values'] = past_key_values
            
            if self.accepts_start_pos:
                model_inputs['start_pos'] = start_pos

            outputs = self.model(**model_inputs)

            if isinstance(outputs, (tuple, list)):
                logits = outputs[0]
                if use_cache and len(outputs) > 1:
                    past_key_values = outputs[1]
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
                if use_cache and hasattr(outputs, 'past_key_values'):
                    past_key_values = outputs.past_key_values
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
            start_pos = curr_input_ids.shape[1] - 1 if use_cache else 0

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
