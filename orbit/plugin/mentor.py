import numpy as np
import json
import os
import locale
from typing import Optional, List, TYPE_CHECKING, Tuple, Dict
from rich.panel import Panel
from orbit.callback import Callback

if TYPE_CHECKING:
    from orbit.engine import Engine
    from orbit.plugin.warmup import Warmup

class Mentor(Callback):
    """
    Mentor 插件：像导师一样监控训练过程，提供改进建议。
    主要关注 Loss 的异常行为（停滞、发散、过拟合、震荡）并结合 Warmup 状态给出建议。
    """
    def __init__(
        self,
        patience: int = 3,
        threshold: float = 1e-4,
        divergence_threshold: float = 2.0,
        language: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Args:
            patience (int): 连续多少个 Epoch Loss 没有显著改善视为停滞。
            threshold (float): 视为改善的最小 Loss 变化量。
            divergence_threshold (float): 当前 Loss 是最小 Loss 的多少倍视为发散。
            language (str): 语言代码 ('en' 或 'zh')。如果为 None，则自动检测系统语言。
            verbose (bool): 是否打印建议。
        """
        super().__init__()
        self.patience = patience
        self.threshold = threshold
        self.divergence_threshold = divergence_threshold
        self.language = language if language else self._detect_language()
        self.verbose = verbose
        
        self.loss_history: List[float] = []
        self.val_loss_history: List[float] = []
        self.min_loss = np.inf
        self.stagnation_counter = 0
        self.increase_counter = 0
        
        # 记录 Warmup 相关信息
        self.has_warmup = False
        self.warmup_plugin: Optional['Warmup'] = None
        
        # 记录其他配置
        self.has_scheduler = False
        self.grad_clip_norm = None
        self.batch_size = 0
        self.accum_steps = 1
        
        # 加载翻译
        self.i18n = self._load_i18n()

    def _detect_language(self) -> str:
        try:
            lang_code, _ = locale.getdefaultlocale()
            if lang_code and lang_code.startswith('zh'):
                return 'zh'
        except:
            pass
        return 'en'

    def _load_i18n(self) -> Dict[str, str]:
        try:
            # 获取当前文件所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(current_dir, 'data', 'mentor_i18n.json')
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            return data.get(self.language, data.get('en', {}))
        except Exception as e:
            print(f"[Mentor] Warning: Failed to load i18n file: {e}. Fallback to keys.")
            return {}

    def _t(self, key: str, **kwargs) -> str:
        """获取翻译并格式化"""
        template = self.i18n.get(key, key)
        try:
            return template.format(**kwargs)
        except KeyError:
            return template

    def on_train_start(self, engine: 'Engine'):
        # 1. 检查 Warmup
        for p in engine.plugins:
            if p.__class__.__name__ == 'Warmup':
                self.has_warmup = True
                self.warmup_plugin = p
                break
        
        # 2. 检查 Scheduler & Gradient Clipping
        self.has_scheduler = engine.scheduler is not None
        self.grad_clip_norm = getattr(engine, 'grad_clip_norm', None)
        
        # 3. 检查 Batch Size & Accumulation
        self.accum_steps = engine.accumulation_steps
        if hasattr(engine, 'train_loader') and hasattr(engine.train_loader, 'batch_size'):
            self.batch_size = engine.train_loader.batch_size or 1 # Handle None
        
        if self.verbose:
            eff_bs = self.batch_size * self.accum_steps
            msg = self._t("mentor_watching", eff_bs=eff_bs, bs=self.batch_size, accum=self.accum_steps)
            engine.print(msg, plugin='Mentor')

    def on_epoch_end(self, engine: 'Engine'):
        current_loss = engine.metrics.get('train_loss')
        val_loss = engine.metrics.get('val_loss')
        
        if current_loss is None:
            return

        self.loss_history.append(current_loss)
        if val_loss is not None:
            self.val_loss_history.append(val_loss)
        
        # 收集本 Epoch 的所有建议 (Title, Content, Style)
        advice_list: List[Tuple[str, str, str]] = []

        # 1. 检查 NaN / Inf
        if not np.isfinite(current_loss):
            advice_list.append((
                self._t("nan_loss_title"),
                self._t("nan_loss_msg"),
                "bold red"
            ))
        else:
            # 更新最小 Loss 和 连续上升计数
            if current_loss < self.min_loss:
                self.min_loss = current_loss
                self.stagnation_counter = 0
                self.increase_counter = 0
            else:
                self.stagnation_counter += 1
                # 检查是否比上一个 epoch 增加
                if len(self.loss_history) > 1 and current_loss > self.loss_history[-2]:
                    self.increase_counter += 1
                else:
                    self.increase_counter = 0

            # 2. 检查发散 (Divergence)
            is_diverging = (current_loss > self.min_loss * self.divergence_threshold)
            is_unstable = (self.increase_counter >= 2)
            
            if (is_diverging or is_unstable) and len(self.loss_history) > 1:
                res = self._analyze_divergence(engine, current_loss, is_unstable)
                if res: advice_list.append(res)
                
                # 重置以避免每个 epoch 都刷屏
                if is_diverging:
                    self.min_loss = current_loss 
                if is_unstable:
                    self.increase_counter = 0

            # 3. 检查停滞 (Stagnation)
            if self.stagnation_counter >= self.patience:
                res = self._analyze_stagnation(engine, current_loss)
                if res: advice_list.append(res)
                self.stagnation_counter = 0 # 重置计数器

            # 4. 检查过拟合 (Overfitting)
            if len(self.val_loss_history) >= 3:
                res = self._analyze_overfitting(engine)
                if res: advice_list.append(res)

            # 5. 检查震荡 (Oscillation)
            if len(self.loss_history) >= 5:
                res = self._analyze_oscillation(engine)
                if res: advice_list.append(res)

        # 统一打印
        if advice_list and self.verbose:
            with engine.out_logs:
                for title, content, style in advice_list:
                    engine.console.print(Panel(content, title=title, border_style=style, expand=False))

    def _analyze_divergence(self, engine: 'Engine', current_loss: float, is_unstable: bool = False) -> Tuple[str, str, str]:
        in_warmup = engine.is_in_warmup()
        eff_bs = self.batch_size * self.accum_steps
        
        title = self._t("divergence_title")
        if is_unstable:
            msg = self._t("divergence_msg_unstable", count=self.increase_counter, loss=current_loss)
        else:
            msg = self._t("divergence_msg_spike", loss=current_loss, min_loss=self.min_loss)
        
        advice = []
        
        # 1. Warmup 建议
        if not self.has_warmup:
            advice.append(self._t("advice_add_warmup"))
            advice.append(self._t("advice_lower_lr"))
        elif in_warmup:
            advice.append(self._t("advice_warmup_start_lr"))
        else:
            advice.append(self._t("advice_post_warmup_lr"))
        
        # 2. Batch Size & Accumulation 建议
        if eff_bs < 32:
            advice.append(self._t("advice_small_bs", eff_bs=eff_bs))
            advice.append(self._t("advice_increase_accum", accum_steps=self.accum_steps))
        
        # 3. 通用建议
        advice.append(self._t("advice_grad_clip"))
            
        return title, msg + "\n\n" + "\n".join(advice), "red"

    def _analyze_stagnation(self, engine: 'Engine', current_loss: float) -> Tuple[str, str, str]:
        in_warmup = engine.is_in_warmup()
        eff_bs = self.batch_size * self.accum_steps
        
        title = self._t("stagnation_title")
        msg = self._t("stagnation_msg", patience=self.patience)
        
        advice = []
        
        if in_warmup:
            # 在 Warmup 期间停滞
            warmup_epochs = getattr(self.warmup_plugin, 'warmup_epochs', 0)
            total_epochs = engine.num_epochs
            
            advice.append(self._t("advice_warmup_duration", epoch=engine.epoch+1))
            if warmup_epochs > total_epochs * 0.2:
                advice.append(self._t("advice_warmup_too_long", warmup_epochs=warmup_epochs))
            advice.append(self._t("advice_check_start_lr"))
            
        else:
            # 非 Warmup 期间停滞
            advice.append(self._t("advice_lr_general"))
            
            # Scheduler 建议
            if not self.has_scheduler:
                advice.append(self._t("advice_add_scheduler"))
            else:
                advice.append(self._t("advice_check_scheduler"))
            
            # Batch Size 建议
            if eff_bs > 4096:
                advice.append(self._t("advice_large_bs", eff_bs=eff_bs))
                advice.append(self._t("advice_reduce_bs"))

            # 4. 初始化建议 (仅在早期)
            if engine.epoch < 10:
                advice.append(self._t("advice_check_init"))

            # 5. 调试与数据建议
            advice.append(self._t("advice_overfit_single_batch"))
            advice.append(self._t("advice_data_hard"))

        return title, msg + "\n\n" + "\n".join(advice), "yellow"

    def _analyze_overfitting(self, engine: 'Engine') -> Optional[Tuple[str, str, str]]:
        """检测过拟合：Train Loss 下降，Val Loss 上升"""
        if len(self.val_loss_history) < 3 or len(self.loss_history) < 3:
            return None
            
        # 检查最近 3 个 epoch
        recent_val = self.val_loss_history[-3:]
        recent_train = self.loss_history[-3:]
        
        # Val Loss 持续上升
        val_rising = (recent_val[-1] > recent_val[-2] > recent_val[-3])
        # Train Loss 持续下降 (或保持低位)
        train_dropping = (recent_train[-1] <= recent_train[-2])
        
        if val_rising and train_dropping:
            title = self._t("overfitting_title")
            msg = self._t("overfitting_msg")
            advice = [
                self._t("advice_regularization"),
                self._t("advice_data_aug"),
                self._t("advice_early_stopping")
            ]
            return title, msg + "\n\n" + "\n".join(advice), "magenta"
        return None

    def _analyze_oscillation(self, engine: 'Engine') -> Optional[Tuple[str, str, str]]:
        """检测震荡：Loss 标准差过大"""
        if len(self.loss_history) < 5:
            return None
            
        recent_loss = self.loss_history[-5:]
        std_dev = np.std(recent_loss)
        mean_loss = np.mean(recent_loss)
        
        # 如果标准差超过均值的 10% (经验值)，且没有持续下降趋势
        # 简单的趋势检查：首尾差异不大
        is_flat_trend = abs(recent_loss[-1] - recent_loss[0]) < std_dev
        
        if std_dev > 0.1 * mean_loss and is_flat_trend:
            title = self._t("oscillation_title")
            msg = self._t("oscillation_msg", std=std_dev)
            advice = [
                self._t("advice_lower_lr_oscillation")
            ]
            
            if not self.has_scheduler:
                advice.append(self._t("advice_oscillation_scheduler"))
            
            if not self.grad_clip_norm:
                advice.append(self._t("advice_oscillation_grad_clip"))
                
            return title, msg + "\n\n" + "\n".join(advice), "cyan"
        return None
