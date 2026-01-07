from transformers import AutoModelForCausalLM, AutoTokenizer
from orbit.kit import ChatInterface
from orbit.utils import inject_lora_file

model_id = 'Qwen/Qwen2.5-0.5B-Instruct' 

print(f'正在加载模型: {model_id} ...')
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    dtype='auto', 
    device_map='auto'
)

inject_lora_file(model, '', merge_and_unload=True, verbose=True)

interface = ChatInterface(model=model, tokenizer=tokenizer, model_role='assistant')

user_input = '你好，请用两句话介绍一下什么是量子力学。'

messages = [
    {'role': 'user', 'content': user_input}
]

print(f'\n用户: {user_input}\n')
print('AI: ', end='')

for chunk in interface.stream_chat(messages):
    print(chunk, end='', flush=True)

print('\n\n[生成结束]')
