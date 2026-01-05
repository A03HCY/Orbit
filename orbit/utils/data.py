from typing import Dict, Any

def build_sft(
    user_content: str, 
    model_content: str, 
    tokenizer: Any, 
    max_length: int = 2048
) -> Dict[str, Any]:
    '''构建 SFT (Supervised Fine-Tuning) 数据集样本。

    将用户输入和模型输出组合，应用对话模板，并进行分词、截断和 padding。
    同时生成 labels，其中用户输入部分和 padding 部分被 mask (设置为 -100)。

    Args:
        user_content (str): 用户的输入内容。
        model_content (str): 模型的期望回复内容。
        tokenizer (Any): 分词器实例，需要支持 apply_chat_template 和 __call__。
        max_length (int, optional): 序列最大长度。默认为 2048。

    Returns:
        Dict[str, Any]: 包含处理后的数据字典：
            - 'input_ids': 输入 token ID 张量。
            - 'attention_mask': 注意力掩码张量。
            - 'labels': 用于计算损失的标签张量，用户部分已 mask。
    '''
    messages = [
        {'role': 'user', 'content': user_content},
        {'role': 'model', 'content': model_content}
    ]

    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )

    encodings = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    input_ids = encodings['input_ids'][0]
    attention_mask = encodings['attention_mask'][0]

    labels = input_ids.clone()

    user_messages = [{'role': 'user', 'content': user_content}]
    user_text = tokenizer.apply_chat_template(
        user_messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    user_encodings = tokenizer(
        user_text, 
        truncation=True, 
        max_length=max_length,
        add_special_tokens=True
    )
    user_len = len(user_encodings['input_ids'])

    if user_len > len(input_ids):
        user_len = len(input_ids)

    labels[:user_len] = -100

    labels[input_ids == tokenizer.pad_token_id] = -100

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
