import math
import warnings
import re
import torch
import torch.nn as nn
from torch.nn.init import _calculate_fan_in_and_fan_out

try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    '''截断正态分布初始化的辅助函数，在无梯度模式下运行。
    
    Args:
        tensor (torch.Tensor): 要初始化的张量。
        mean (float): 正态分布的均值。
        std (float): 正态分布的标准差。
        a (float): 截断下界。
        b (float): 截断上界。
        
    Returns:
        torch.Tensor: 初始化后的张量。
    '''
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    '''使用截断正态分布填充输入张量。
    
    Args:
        tensor (torch.Tensor): 要填充的 n 维 torch.Tensor。
        mean (float): 正态分布的均值。
        std (float): 正态分布的标准差。
        a (float): 最小截止值。
        b (float): 最大截止值。
        
    Returns:
        torch.Tensor: 修改后的张量。
    '''
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def constant_init(module, val, bias=0):
    '''使用常数值初始化模块权重，可选初始化偏置。
    
    Args:
        module (nn.Module): 要初始化的模块。
        val (float): 权重的常数值。
        bias (float): 偏置的常数值。
    '''
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def init_weights(module, method='kaiming', distribution='normal', bias=0, 
                 a=0, mode='fan_out', nonlinearity='relu', 
                 gain=1, 
                 std=0.02, trunc_a=-2., trunc_b=2.):
    '''通用权重初始化函数，支持多种初始化方法。
    
    Args:
        module (nn.Module): 要初始化的模块。
        method (str): 初始化方法。可选值：
            - 'kaiming': Kaiming (He) 初始化
            - 'xavier': Xavier (Glorot) 初始化
            - 'c2_xavier': Caffe2 风格的 Xavier 初始化
            - 'orthogonal': 正交初始化
            - 'trunc_normal': 截断正态分布
            - 'normal': 标准正态分布
            - 'constant': 常数初始化 (使用 val=gain)
        distribution (str): 'uniform' 或 'normal' (用于 kaiming 和 xavier)。
        bias (float): 偏置的初始化值。
        a (float): Kaiming init 的负斜率。
        mode (str): Kaiming init 的模式 ('fan_in', 'fan_out')。
        nonlinearity (str): Kaiming init 的非线性函数 ('relu', 'leaky_relu' 等)。
        gain (float): Xavier init 的缩放因子，或 Constant init 的值。
        std (float): Normal/Truncated Normal 的标准差。
        trunc_a (float): Truncated Normal 的下界。
        trunc_b (float): Truncated Normal 的上界。
    '''
    if hasattr(module, 'weight') and module.weight is not None:
        if method == 'kaiming':
            if distribution == 'uniform':
                nn.init.kaiming_uniform_(
                    module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
            else:
                nn.init.kaiming_normal_(
                    module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        
        elif method == 'xavier':
            if distribution == 'uniform':
                nn.init.xavier_uniform_(module.weight, gain=gain)
            else:
                nn.init.xavier_normal_(module.weight, gain=gain)
        
        elif method == 'c2_xavier':
            fan_in, fan_out = _calculate_fan_in_and_fan_out(module.weight)
            c2_std = math.sqrt(1.0 / float(fan_in))
            nn.init.normal_(module.weight, mean=0.0, std=c2_std)
            
        elif method == 'orthogonal':
            nn.init.orthogonal_(module.weight, gain=gain)
            
        elif method == 'trunc_normal':
            trunc_normal_(
                module.weight, mean=0., std=std, a=trunc_a, b=trunc_b)
        
        elif method == 'normal':
            nn.init.normal_(module.weight, mean=0., std=std)
            
        elif method == 'constant':
            nn.init.constant_(module.weight, val=gain)
            
        else:
            nn.init.xavier_uniform_(module.weight, gain=gain)

    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def init_layer_norm(module, weight=1.0, bias=0.0):
    '''初始化 LayerNorm 或 GroupNorm 模块。
    
    Args:
        module (nn.Module): 归一化模块。
        weight (float): 权重的初始值 (gamma)。
        bias (float): 偏置的初始值 (beta)。
    '''
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, weight)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def init_embedding(module, init_method='normal', std=0.02, a=0., b=1., padding_idx=None):
    '''初始化 Embedding 层。
    
    Args:
        module (nn.Embedding): Embedding 模块。
        init_method (str): 'normal', 'trunc_normal', 'uniform'。
        std (float): 正态分布的标准差。
        a (float): 均匀分布的下界或截断正态分布的下界。
        b (float): 均匀分布的上界或截断正态分布的上界。
        padding_idx (int, optional): 如果指定，padding 索引的权重将被初始化为 0。
    '''
    if hasattr(module, 'weight') and module.weight is not None:
        if init_method == 'normal':
            nn.init.normal_(module.weight, mean=0., std=std)
        elif init_method == 'trunc_normal':
            trunc_normal_(module.weight, mean=0., std=std, a=a, b=b)
        elif init_method == 'uniform':
            nn.init.uniform_(module.weight, a=a, b=b)
    
    if padding_idx is not None:
        module.weight.data[padding_idx].zero_()

def init_weights_transformer(model, n_layer=None, initializer_range=0.02, 
                             residual_proj_names=('linear_out', 'fc2', 'c_proj'),
                             verbose=False):
    '''Transformer 模型的通用初始化逻辑，支持复杂的嵌套结构和残差缩放。
    
    Args:
        model (nn.Module): 要初始化的模型。
        n_layer (int, optional): Transformer 的层数，用于残差缩放。
        initializer_range (float): 初始化的标准差。
        residual_proj_names (tuple): 需要应用残差缩放的模块名称关键字。
        verbose (bool): 是否打印初始化信息。
    '''
    init_info = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            
            info = f'Normal (std={initializer_range})'
            if n_layer is not None:
                module_name = name.split('.')[-1]
                if any(proj_name in module_name for proj_name in residual_proj_names):
                    scale = 1.0 / math.sqrt(2.0 * n_layer)
                    module.weight.data.mul_(scale)
                    info = f'Residual Scaled (scale={scale:.4f})'
            
            init_info.append((name, module.__class__.__name__, info))
        
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            if hasattr(module, 'padding_idx') and module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
            init_info.append((name, 'Embedding', f'Normal (std={initializer_range})'))
        
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
            init_info.append((name, module.__class__.__name__, 'Ones/Zeros'))

    if verbose:
        _print_init_info(init_info)

class WeightInitializer:
    '''多功能权重初始化器类，支持复杂的初始化策略配置。
    
    Attributes:
        method (str): 主要初始化方法。
        distribution (str): 分布类型。
        init_bias (float): 偏置初始值。
        init_norm_weight (float): 归一化层权重初始值。
        init_norm_bias (float): 归一化层偏置初始值。
    '''

    def __init__(
        self,
        method='kaiming',
        distribution='normal',
        mode='fan_out',
        nonlinearity='relu',
        init_bias=0.0,
        init_norm_weight=1.0,
        init_norm_bias=0.0,
        std=0.02,
        trunc_a=-2.0,
        trunc_b=2.0
    ):
        '''初始化 WeightInitializer。
        
        Args:
            method (str): 初始化方法，默认为 'kaiming'。
            distribution (str): 'uniform' 或 'normal'。
            mode (str): 'fan_in' 或 'fan_out' (用于 kaiming)。
            nonlinearity (str): 非线性函数名 (用于 kaiming)。
            init_bias (float): 线性/卷积层的偏置初始值。
            init_norm_weight (float): Norm 层的权重初始值。
            init_norm_bias (float): Norm 层的偏置初始值。
            std (float): 用于 'normal' 或 'trunc_normal' 的标准差。
            trunc_a (float): 截断正态分布下界。
            trunc_b (float): 截断正态分布上界。
        '''
        self.method = method
        self.distribution = distribution
        self.mode = mode
        self.nonlinearity = nonlinearity
        self.init_bias = init_bias
        self.init_norm_weight = init_norm_weight
        self.init_norm_bias = init_norm_bias
        self.std = std
        self.trunc_a = trunc_a
        self.trunc_b = trunc_b

    def apply(self, model, override=None, verbose=False):
        '''将初始化策略应用于模型的所有子模块。
        
        Args:
            model (nn.Module): 要初始化的模型。
            override (dict, optional): 针对特定层名称的覆盖配置。
                格式: {'regex_pattern': {'method': '...', ...}}
            verbose (bool): 是否打印初始化信息。
        '''
        init_info = []

        for name, module in model.named_modules():
            current_config = {}
            if override:
                for pattern, config in override.items():
                    if re.search(pattern, name):
                        current_config = config
                        break
            
            method = current_config.get('method', self.method)
            distribution = current_config.get('distribution', self.distribution)
            mode = current_config.get('mode', self.mode)
            nonlinearity = current_config.get('nonlinearity', self.nonlinearity)
            std = current_config.get('std', self.std)
            
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Conv3d, nn.Linear)):
                init_weights(
                    module,
                    method=method,
                    distribution=distribution,
                    bias=self.init_bias,
                    mode=mode,
                    nonlinearity=nonlinearity,
                    std=std,
                    trunc_a=self.trunc_a,
                    trunc_b=self.trunc_b
                )
                info = f'{method} ({distribution})'
                if method == 'kaiming':
                    info += f', mode={mode}, nonlin={nonlinearity}'
                elif method == 'normal':
                    info += f', std={std}'
                init_info.append((name, module.__class__.__name__, info))
            
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm1d)):
                init_layer_norm(
                    module, weight=self.init_norm_weight, bias=self.init_norm_bias)
                init_info.append((name, module.__class__.__name__, 'Norm (1/0)'))
            
            elif isinstance(module, nn.Embedding):
                init_embedding(
                    module, 
                    init_method=method if method in ['normal', 'trunc_normal', 'uniform'] else 'normal', 
                    std=std
                )
                init_info.append((name, 'Embedding', f'{method} (std={std})'))

        if verbose:
            _print_init_info(init_info)

def _print_init_info(init_info):
    '''打印初始化信息的辅助函数，支持 rich 美化。
    
    Args:
        init_info (list): 包含 (layer_name, module_type, details) 元组的列表。
    '''
    if not init_info:
        return

    if RICH_AVAILABLE:
        console = Console()
        table = Table(title="Weight Initialization Report", show_header=True, header_style="bold magenta")
        table.add_column("Layer Name", style="cyan")
        table.add_column("Module Type", style="green")
        table.add_column("Initialization Details", style="yellow")

        for name, type_name, details in init_info:
            table.add_row(name, type_name, details)
        
        console.print(table)
    else:
        print(f"{'Layer Name':<40} | {'Module Type':<20} | {'Initialization Details'}")
        print("-" * 90)
        for name, type_name, details in init_info:
            print(f"{name:<40} | {type_name:<20} | {details}")

def initialize_weights(model, method='kaiming', override=None, verbose=False, **kwargs):
    '''初始化模型权重的便捷函数。
    
    Args:
        model (nn.Module): 要初始化的模型。
        method (str): 初始化方法。
        override (dict, optional): 针对特定层名称的覆盖配置。
        verbose (bool): 是否打印初始化信息。
        **kwargs: 传递给 WeightInitializer 的其他参数。
    '''
    initializer = WeightInitializer(method=method, **kwargs)
    initializer.apply(model, override=override, verbose=verbose)

class AutoInitializer:
    '''自动初始化器，通过分析模型结构统计信息来应用最优初始化策略。
    
    该类会自动探测模型的深度、激活函数分布以及是否包含 Transformer 结构，
    并据此推荐合适的初始化方法（如 Kaiming, Xavier, 或带残差缩放的正态分布）。
    
    Attributes:
        model (nn.Module): 需要初始化的模型。
        stats (dict): 模型分析统计信息。
    '''
    
    def __init__(self, model):
        '''初始化 AutoInitializer。
        
        Args:
            model (nn.Module): 需要初始化的模型。
        '''
        self.model = model
        self.stats = self._analyze_model()
        
    def _analyze_model(self):
        '''分析模型结构，收集统计信息。
        
        Returns:
            dict: 包含深度、激活函数分布、层类型分布等信息的字典。
        '''
        stats = {
            'depth': 0,
            'activations': {},
            'layer_types': {},
            'transformer_detected': False
        }
        
        # 简单的深度估计：计算包含参数的层数
        param_layers = [m for m in self.model.modules() if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d))]
        stats['depth'] = len(param_layers)
        
        # 激活函数探测
        for m in self.model.modules():
            name = m.__class__.__name__
            if 'ReLU' in name:
                stats['activations']['relu'] = stats['activations'].get('relu', 0) + 1
            elif 'GELU' in name:
                stats['activations']['gelu'] = stats['activations'].get('gelu', 0) + 1
            elif 'Tanh' in name:
                stats['activations']['tanh'] = stats['activations'].get('tanh', 0) + 1
            elif 'Sigmoid' in name:
                stats['activations']['sigmoid'] = stats['activations'].get('sigmoid', 0) + 1
            
            # Transformer 检测
            if 'Attention' in name or 'Transformer' in name:
                stats['transformer_detected'] = True
                
            stats['layer_types'][name] = stats['layer_types'].get(name, 0) + 1
            
        return stats

    def recommend_config(self):
        '''基于统计信息推荐初始化配置。
        
        Returns:
            tuple: (method, nonlinearity, config)
                - method (str): 推荐的主要初始化方法。
                - nonlinearity (str): 推荐的非线性函数参数。
                - config (dict): 针对特定层的覆盖配置。
        '''
        config = {}
        method = 'kaiming' # 默认
        nonlinearity = 'relu'
        
        # 1. 确定主要激活函数
        acts = self.stats['activations']
        if acts:
            main_act = max(acts, key=acts.get)
            if main_act == 'relu':
                method = 'kaiming'
                nonlinearity = 'relu'
            elif main_act == 'gelu':
                method = 'trunc_normal' # GELU 通常配合正态分布
            elif main_act in ['tanh', 'sigmoid']:
                method = 'xavier'
        
        # 2. Transformer 特殊处理
        if self.stats['transformer_detected']:
            # 对于 Transformer，通常使用正态分布
            method = 'normal'
            # 残差缩放配置
            n_layers = max(1, self.stats['depth'] // 4) # 粗略估计 Block 数量
            scale = 1.0 / math.sqrt(2.0 * n_layers)
            
            # 针对投影层的覆盖配置
            config['.*linear_out.*'] = {'method': 'normal', 'std': 0.02 * scale}
            config['.*fc2.*'] = {'method': 'normal', 'std': 0.02 * scale}
            config['.*c_proj.*'] = {'method': 'normal', 'std': 0.02 * scale}
            
        return method, nonlinearity, config

    def apply(self, verbose=True):
        '''应用自动生成的初始化策略。
        
        Args:
            verbose (bool): 是否打印分析报告和初始化详情。
        '''
        method, nonlinearity, override = self.recommend_config()
        
        if verbose:
            if RICH_AVAILABLE:
                console = Console()
                console.print(f"[bold cyan]Auto Initialization Analysis:[/bold cyan]")
                console.print(f"  Depth: {self.stats['depth']}")
                console.print(f"  Activations: {self.stats['activations']}")
                console.print(f"  Transformer Detected: {self.stats['transformer_detected']}")
                console.print(f"[bold green]Recommended Strategy:[/bold green] {method} (nonlin={nonlinearity})")
            else:
                print(f"Auto Init: Depth={self.stats['depth']}, Acts={self.stats['activations']}")
                print(f"Strategy: {method}, {nonlinearity}")

        initialize_weights(
            self.model, 
            method=method, 
            nonlinearity=nonlinearity, 
            override=override, 
            verbose=verbose
        )

def auto_initialize(model, verbose=True):
    '''自动初始化的便捷入口函数。
    
    Args:
        model (nn.Module): 需要初始化的模型。
        verbose (bool): 是否打印初始化信息。
    '''
    initializer = AutoInitializer(model)
    initializer.apply(verbose=verbose)
