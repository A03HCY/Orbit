import math
import torch

# 此代码片段改编自以下 GitHub 仓库的修改版本：
# https://github.com/KellerJordan/Muon/blob/master/muon.py
@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    '''
    使用 Newton-Schulz 迭代计算 G 的零次幂/正交化。我们选择使用五次迭代，
    其系数被选择为最大化零处的斜率。为了最小化步骤，经验表明，即使在迭代不再在区间上的
    所有位置完全收敛到 1 的点之后，继续增加零处的斜率也是有效的。因此，此迭代不产生 UV^T，
    而是产生类似 US'V^T 的结果，其中 S' 是对角矩阵，S_{ii}' ~ Uniform(0.5, 1.5)，
    这在模型性能方面相对于 UV^T（其中 USV^T = G 是 SVD）完全没有负面影响。
    '''
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # 确保谱范数至多为 1
    X = X / (X.norm() + 1e-7)
    # 执行 NS 迭代
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # 改编自 @jxbz, @leloykun 和 @YouJiacheng 的建议
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    '''
    Muon - MomentUm Orthogonalized by Newton-schulz (通过 Newton-Schulz 正交化的动量)

    Muon 内部运行标准的 SGD-momentum，然后执行正交化后处理步骤，
    其中每个 2D 参数的更新都被替换为最近的正交矩阵。为了有效地正交化每个更新，
    我们使用 Newton-Schulz 迭代，其优点是可以在 GPU 上以 bfloat16 稳定运行。

    一些警告：
    - 我们认为此优化器不太可能在小批量训练中表现良好。
    - 我们认为它可能不适合微调预训练模型，但我们尚未对此进行测试。

    参数:
        muon_params: 要由 Muon 优化的参数。
        lr: 学习率。更新的谱范数将为 `lr`。（0.02 是一个很好的默认值）
        momentum: 内部 SGD 使用的动量。（0.95 是一个很好的默认值）
        nesterov: 是否在内部 SGD 中使用 Nesterov 风格的动量。（推荐）
        ns_steps: 要运行的 Newton-Schulz 迭代次数。（6 可能总是足够的）
        adamw_params: 要由 AdamW 优化的参数。`muon_params` 中任何 {0, 1}-D 参数
            或被检测为嵌入或 lm_head 的参数也将由 AdamW 优化。
        adamw_lr: 内部 AdamW 的学习率。
        adamw_betas: 内部 AdamW 的 beta 参数。
        adamw_eps: 内部 AdamW 的 epsilon 参数。
        adamw_wd: 内部 AdamW 的权重衰减。
    '''

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # 将参数分类为我们将使用 Muon 的参数和不使用的参数
        for p in muon_params:
            # 对 muon_params 中每个 >= 2D 且看起来不像嵌入或头层的参数使用 Muon
            assert p.ndim == 2, p.ndim
            self.state[p]['use_muon'] = True
        for p in adamw_params:
            # 对 adamw_params 中的参数不使用 Muon
            self.state[p]['use_muon'] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # 我们根据参数矩阵的大小调整学习率和权重衰减，如论文中所述
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        '''
        执行单个优化步骤。

        Args:
            closure (Callable, optional): 重新评估模型并返回损失的闭包。
        '''
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group['params'] if self.state[p]['use_muon']]
            # import pdb; pdb.set_trace()
            lr = group['lr']
            wd = group['wd']
            momentum = group['momentum']

            # 生成权重更新
            for p in params:
                # 完整性检查
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # 计算更新
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                u = zeropower_via_newtonschulz5(g, steps=group['ns_steps'])

                # 缩放更新
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # 应用权重衰减
                p.data.mul_(1 - lr * wd)

                # 应用更新
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group['params'] if not self.state[p]['use_muon']]
            lr = group['lr']
            beta1, beta2 = group['adamw_betas']
            eps = group['adamw_eps']
            weight_decay = group['wd']

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                    state['moment1'] = torch.zeros_like(g)
                    state['moment2'] = torch.zeros_like(g)
                state['step'] += 1
                step = state['step']
                buf1 = state['moment1']
                buf2 = state['moment2']
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss
