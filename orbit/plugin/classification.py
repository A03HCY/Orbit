import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from rich.table import Table
from typing import List, Optional, TYPE_CHECKING

from orbit.callback import Callback
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

    def on_eval_start(self, engine: "Engine"):
        """每轮验证开始前清空缓存"""
        self.preds = []
        self.targets = []

    def on_batch_end(self, engine: "Engine"):
        """收集验证阶段的预测结果"""
        if engine.state == "EVAL":
            # 假设 engine.output 是 logits [Batch, NumClasses]
            # 假设 engine.target 是 labels [Batch]
            
            # 收集 Raw Output (用于 Top-K) 或 Argmax (用于混淆矩阵)
            # 为了节省内存，我们这里尽量存 CPU Tensor
            self.preds.append(engine.output.detach().cpu()) 
            self.targets.append(engine.target.detach().cpu())

    def on_eval_end(self, engine: "Engine"):
        """验证结束后计算指标并绘图"""
        if not self.preds: return

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
        
        # --- B. 控制台打印 Classification Report ---
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names, 
            output_dict=True,
            zero_division=0
        )
        self._print_rich_table(engine, report, acc)

        # --- C. 绘制 Confusion Matrix ---
        # 只有挂载了 TensorBoard Writer 才画图
        if hasattr(engine, 'writer') and engine.writer is not None:
            fig = self._plot_confusion_matrix(y_true, y_pred)
            engine.writer.add_figure("Eval/Confusion_Matrix", fig, global_step=engine.epoch)
            plt.close(fig) # 关闭 release 内存

    def _print_rich_table(self, engine, report: dict, acc: float):
        """用 Rich 打印漂亮的分类报告表格"""
        table = Table(title=f"[bold]Evaluation Report (Ep {engine.epoch+1})[/]")
        table.add_column("Class", style="cyan")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("F1-Score", justify="right")

        for class_name in self.class_names:
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
        
        engine.print(table)
        engine.print(f"[green]Accuracy: {acc*100:.2f}%[/]")

    def _plot_confusion_matrix(self, y_true, y_pred):
        """使用 Seaborn 绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        # 创建 Figure
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap=self.cm_cmap,
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        return fig
