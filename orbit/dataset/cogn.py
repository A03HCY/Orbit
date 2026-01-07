import os
import json
from typing import Any, Dict
from dataclasses import dataclass, asdict
from torch.utils.data import Dataset
from orbit.utils import build_sft

@dataclass
class CognField:
    '''
    自我认知字段定义
    '''
    model_name: str = 'AI'
    model_developer: str = 'AI Team'
    model_version: str = '1.0'
    knowledge_cutoff: str = '2024.1'
    capabilities: str = '逻辑推理、文本生成、代码编写和语言翻译'
    multimodal_support: str = '文本'
    limitations: str = '无法实时连接互联网获取最新资讯, 没有物理躯体或情感体验'
    identity_restriction: str = '人工智能语言模型'

class CognDataset(Dataset):
    '''
    自我认知数据集类
    '''
    def __init__(self, data: list[dict[str, str]]):
        '''
        初始化数据集

        Args:
            data (list[dict[str, str]]): 包含指令和响应的字典列表 instruction 与 response
        '''
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def build_self_cogn(zh: CognField, en: CognField) -> CognDataset:
    '''
    构建自我认知数据集

    Args:
        zh (CognField): 中文模型信息对象
        en (CognField): 英文模型信息对象

    Returns:
        CognDataset: Torch 数据集对象
    '''
    data = []
    base_dir = os.path.join(os.path.dirname(__file__), 'data')

    zh_path = os.path.join(base_dir, 'cogn_zh.jsonl')
    zh_dict = asdict(zh)
    if os.path.exists(zh_path):
        with open(zh_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                response = item['response']
                for key, value in zh_dict.items():
                    response = response.replace(f'{{{{{key}}}}}', value)
                item['response'] = response
                data.append(item)

    en_path = os.path.join(base_dir, 'cogn_en.jsonl')
    en_dict = asdict(en)
    if os.path.exists(en_path):
        with open(en_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                response = item['response']
                for key, value in en_dict.items():
                    response = response.replace(f'{{{{{key}}}}}', value)
                item['response'] = response
                data.append(item)

    return CognDataset(data)

class CognitionSFT(Dataset):
    '''自我认知 SFT 数据集包装类，将原始文本转换为训练张量。'''

    def __init__(
        self, 
        tokenizer: Any, 
        zh_field: CognField, 
        en_field: CognField, 
        max_length: int = 2048,
        model_role: str = 'model',
        padding: bool = True,
        ignore_index: int = -100
    ):
        '''初始化自我认知 SFT 数据集。

        Args:
            tokenizer (Any): 分词器实例。
            zh_field (CognField): 中文模型信息配置。
            en_field (CognField): 英文模型信息配置。
            max_length (int, optional): 序列最大长度。默认为 2048。
            model_role (str, optional): 模型角色名称。默认为 'model'。
            padding (bool, optional): 是否进行 padding 到 max_length。默认为 True。
            ignore_index (int, optional): 用于 mask labels 的索引值。默认为 -100。
        '''
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_role = model_role
        self.padding = padding
        self.ignore_index = ignore_index
        self.raw_dataset = build_self_cogn(zh_field, en_field)

    def __len__(self) -> int:
        '''获取数据集样本数量。'''
        return len(self.raw_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        '''获取指定索引的 SFT 样本。

        Args:
            index (int): 样本索引。

        Returns:
            Dict[str, Any]: 包含 input_ids, attention_mask 和 labels 的字典。
        '''
        item = self.raw_dataset[index]
        return build_sft(
            user_content=item['instruction'],
            model_content=item['response'],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            model_role=self.model_role,
            padding=self.padding,
            ignore_index=self.ignore_index
        )
