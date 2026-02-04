from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.prompt import Prompt

from .wrapper import AutoRegressiveWrapper
from .token import reasoning_tokens

class ChatInterface:
    '''
    一个用于命令行实时交互的聊天接口类

    Attributes:
        model: 语言模型实例
        tokenizer: 分词器实例
        device: 模型所在的设备
    '''

    def __init__(self, model=None, tokenizer=None, model_id=None, device='auto', dtype='auto', model_role='assistant'):
        '''
        初始化聊天接口

        Args:
            model: 预加载的模型实例。如果为 None，则需要提供 model_id
            tokenizer: 预加载的分词器实例。如果为 None，则需要提供 model_id
            model_id: 模型的 HuggingFace ID 或本地路径
            device: 设备设置，默认为 'auto'
            dtype: 模型的权重精度，默认为 'auto'
            model_role: 模型回复的角色名称，默认为 'assistant'
        '''
        self.console = Console()
        self.model_role = model_role
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        elif model_id is not None:
            self._load_model(model_id, device, dtype)
        else:
            raise ValueError('必须提供 (model 和 tokenizer) 或 model_id')

        self.device = self.model.device

        if not hasattr(self.model, 'generate'):
            self.model = AutoRegressiveWrapper(self.model)

    def _load_model(self, model_id, device, dtype):
        '''
        从指定的 model_id 加载模型和分词器

        Args:
            model_id: 模型的 HuggingFace ID 或本地路径
            device: 设备设置
            dtype: 模型精度
        '''
        with self.console.status(f'[bold green]正在加载模型: {model_id} ...[/bold green]'):
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=dtype,
                device_map=device
            )
        self.console.print(f'[bold green]模型 {model_id} 加载完成！[/bold green]')

    def stream_chat(
        self,
        messages: list,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        enable_thinking: bool = False
    ):
        '''
        流式生成对话响应

        Args:
            messages (list): 符合 ChatML 格式的消息列表
            max_new_tokens (int): 最大新生成的 token 数量
            temperature (float): 生成温度
            top_k (int): Top-k 采样值
            top_p (float): Top-p 采样值
            repetition_penalty (float): 重复惩罚系数
            do_sample (bool): 是否使用采样
            enable_thinking (bool): 是否启用思考模式

        Yields:
            str: 生成的新文本片段
        '''
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        model_inputs = self.tokenizer([text], return_tensors='pt').to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=not enable_thinking
        )

        generation_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            eos_token_id=self.tokenizer.eos_token_id
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            if enable_thinking:
                for special_token in self.tokenizer.all_special_tokens:
                    if special_token not in reasoning_tokens and special_token in new_text:
                        new_text = new_text.replace(special_token, '')
            yield new_text

    def interact(self, enable_thinking: bool = True):
        '''
        启动命令行实时交互会话

        Args:
            enable_thinking (bool): 是否启用思考模式展示
        '''
        self.console.print(Panel(
            '[bold]聊天接口已就绪[/bold]\n输入 [red]"exit"[/red] 或 [red]"quit"[/red] 退出',
            title='[bold blue]Orbit Chat[/bold blue]',
            border_style='blue',
            expand=False
        ))
        history = []
        while True:
            try:
                self.console.print()
                user_input = Prompt.ask('[bold green]User[/bold green]', console=self.console)
                if user_input.lower() in ['exit', 'quit']:
                    break
                if not user_input.strip():
                    continue

                history.append({'role': 'user', 'content': user_input})
                
                self.console.print('[bold purple]Model:[/bold purple]\n')
                full_response = ''
                thought_content = ''
                is_thinking = False
                
                with Live(console=self.console, refresh_per_second=12) as live:
                    for chunk in self.stream_chat(history, enable_thinking=enable_thinking):
                        buffer = chunk
                        
                        if '[cot_start]' in buffer:
                            is_thinking = True
                            parts = buffer.split('[cot_start]')
                            full_response += parts[0]
                            buffer = parts[1]
                        
                        if '[cot_end]' in buffer:
                            is_thinking = False
                            parts = buffer.split('[cot_end]')
                            thought_content += parts[0]
                            buffer = parts[1]
                            
                        if is_thinking:
                            thought_content += buffer
                        else:
                            full_response += buffer
                            
                        renderables = []
                        if thought_content:
                            title = "Thinking..." if is_thinking else "Thought Process"
                            style = "yellow" if is_thinking else "dim"
                            renderables.append(Panel(Markdown(thought_content), title=title, border_style=style))
                            
                        if full_response:
                            renderables.append(Markdown(full_response))
                        
                        if not renderables:
                            renderables.append(Markdown(""))
                            
                        live.update(Group(*renderables))
                
                msg = {'role': self.model_role}
                if thought_content:
                    msg['thought'] = thought_content
                msg['content'] = full_response
                history.append(msg)

            except KeyboardInterrupt:
                self.console.print('\n[bold red][会话已中断][/bold red]')
                break
        
        self.console.print('[bold blue][生成结束][/bold blue]')
