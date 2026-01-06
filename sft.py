from typing import Any
from orbit.dataset import CognField, CognitionSFT

field_zh = CognField(
    model_name='Coplis',
    model_developer='GITM Team',
    model_version='1.0 mini',
    knowledge_cutoff='2024.1',
    capabilities='逻辑推理、文本生成、代码编写和语言翻译',
    multimodal_support='文本',
    limitations='无法实时连接互联网获取最新资讯, 没有物理躯体或情感体验, 不具备多模态理解能力（如图像识别）',
    identity_restriction='人工智能语言模型'
)

field_en = CognField(
    model_name='Coplis',
    model_developer='GITM Team',
    model_version='1.0 mini',
    knowledge_cutoff='January 2024',
    capabilities='logical reasoning, text generation, code writing, and language translation',
    multimodal_support='text',
    limitations='the inability to access the internet in real-time, the lack of a physical body or emotional experiences, and the absence of multimodal understanding (such as image recognition)',
    identity_restriction='an artificial intelligence language model'
)

def get_self_cognition_dataset(tokenizer: Any, max_length: int = 2048) -> CognitionSFT:
    '''便捷函数，用于获取预配置的自我认知数据集。

    Args:
        tokenizer (Any): 分词器实例。
        max_length (int, optional): 序列最大长度。默认为 2048。

    Returns:
        CognitionSFT: 实例化后的数据集对象。
    '''
    return CognitionSFT(
        tokenizer=tokenizer,
        zh_field=field_zh,
        en_field=field_en,
        max_length=max_length
    )
