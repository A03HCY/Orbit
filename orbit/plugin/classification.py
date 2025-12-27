import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from rich.table import Table
from typing import List, Optional, TYPE_CHECKING
import rich.box as box

from orbit.callback import Callback, Event
if TYPE_CHECKING:
    from ..engine import Engine

class ClassificationReport(Callback):
    def __init__(
        self, 
        num_classes: int, 
        class_names: Optional[List[str]] = None,
        top_k: int = 1,
        cm_cmap: str = 'Blues'
    ):
        """
        专用于分类任务的评估与可视化回调。

        Args:
            num_classes (int): 类别总数。
            class_names (List[str]): 类别名称列表 ["Cat", "Dog", ...]。可选。
            top_k (int): 另外计算 Top-K 准确率。
            cm_cmap (str): 混淆矩阵热图的颜色风格。
        """
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names if class_names else [str(i) for i in range(num_classes)]
        self.top_k = top_k
        self.cm_cmap = cm_cmap
        
        # 缓存预测结果
        self.preds = []
        self.targets = []

    def on_eval_start(self, event: Event):
        """每轮验证开始前清空缓存"""
        self.preds = []
        self.targets = []

    def on_batch_end(self, event: Event):
        """收集验证阶段的预测结果"""
        engine = event.engine
        if engine.state == "EVAL":
            # 假设 engine.output 是 logits [Batch, NumClasses]
            # 假设 engine.target 是 labels [Batch]
            
            # 收集 Raw Output (用于 Top-K) 或 Argmax (用于混淆矩阵)
            # 为了节省内存，我们这里尽量存 CPU Tensor
            self.preds.append(engine.output.detach().cpu()) 
            self.targets.append(engine.target.detach().cpu())

    def on_eval_end(self, event: Event):
        """验证结束后计算指标并绘图"""
        if not self.preds: return
        engine = event.engine

        # 1. 拼接所有 Batch
        all_logits = torch.cat(self.preds)  # [N, C]
        all_targets = torch.cat(self.targets) # [N]
        
        # 转为预测类别索引 [N]
        all_preds_idx = all_logits.argmax(dim=1)
        
        # 转换 numpy 用于 sklearn
        y_true = all_targets.numpy()
        y_pred = all_preds_idx.numpy()

        # --- A. 计算基础 Acc 并存入 metrics ---
        acc = accuracy_score(y_true, y_pred)
        engine.metrics['val_acc'] = acc
        
        # --- 计算 Top-K Acc ---
        topk_acc = None
        if self.top_k > 1:
            _, indices = all_logits.topk(self.top_k, dim=1)
            correct = indices.eq(all_targets.view(-1, 1).expand_as(indices))
            topk_acc = correct.sum().item() / len(all_targets)
            engine.metrics[f'val_acc_top{self.top_k}'] = topk_acc

        # --- B. 控制台打印 Classification Report ---
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names, 
            output_dict=True,
            zero_division=0
        )
        self._print_rich_table(engine, report, acc, topk_acc)

        # --- C. 绘制 Confusion Matrix ---
        # 只有挂载了 TensorBoard Writer 才画图
        if hasattr(engine, 'writer') and engine.writer is not None:
            fig = self._plot_confusion_matrix(y_true, y_pred)
            engine.writer.add_figure("Eval/Confusion_Matrix", fig, global_step=engine.epoch)
            plt.close(fig) # 关闭 release 内存

    def _print_rich_table(self, engine, report: dict, acc: float, topk_acc: Optional[float] = None):
        """用 Rich 打印漂亮的分类报告表格"""
        table = Table(title=f"[bold]Evaluation Report (Ep {engine.epoch+1})[/]", box=box.HORIZONTALS)
        table.add_column("Class", style="cyan")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("F1-Score", justify="right")

        # 限制显示数量，防止刷屏
        max_display = 20
        items_to_show = []
        
        if len(self.class_names) > max_display:
            items_to_show.extend(self.class_names[:10])
            items_to_show.append(None) # None 表示省略号
            items_to_show.extend(self.class_names[-10:])
        else:
            items_to_show = self.class_names

        for class_name in items_to_show:
            if class_name is None:
                table.add_row("...", "...", "...", "...")
                continue

            if class_name in report:
                row = report[class_name]
                table.add_row(
                    class_name,
                    f"{row['precision']:.3f}",
                    f"{row['recall']:.3f}",
                    f"{row['f1-score']:.3f}",
                )
        
        avg = report['weighted avg']
        table.add_row(
            "[bold]Weighted Avg[/]",
            f"[bold]{avg['precision']:.3f}[/]",
            f"[bold]{avg['recall']:.3f}[/]",
            f"[bold]{avg['f1-score']:.3f}[/]",
            end_section=True
        )
        
        with engine.out_logs:
            engine.print(table)
        
        engine.print(f"Accuracy: [green]{acc*100:.2f}%[/]", plugin='ClassReport')
        if topk_acc is not None:
            engine.print(f"Top-{self.top_k} Accuracy: [green]{topk_acc*100:.2f}%[/]", plugin='ClassReport')

    def _plot_confusion_matrix(self, y_true, y_pred):
        """使用 Seaborn 绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        num_classes = len(self.class_names)
        
        # 动态调整 Figure 大小
        # 基础大小 10，每多一个类别增加一点尺寸，上限设个限制防止过大
        fig_base = 10
        fig_scale = 0.3
        figsize_dim = max(fig_base, min(50, num_classes * fig_scale))
        
        # 创建 Figure
        fig, ax = plt.subplots(figsize=(figsize_dim, figsize_dim))
        
        # 智能决定是否显示数值 annot
        do_annot = True
        if num_classes > 20:
            do_annot = False

        sns.heatmap(
            cm, 
            annot=do_annot, 
            fmt='d', 
            cmap=self.cm_cmap,
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
            square=True
        )
        
        # 调整标签样式
        if num_classes > 20:
            plt.xticks(rotation=90, fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix ({num_classes} classes)')
        plt.tight_layout()
        return fig
