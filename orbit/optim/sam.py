import torch

class SAM(torch.optim.Optimizer):
    '''锐度感知最小化优化器 (Sharpness-Aware Minimization)。

    该优化器通过寻找损失景观中平坦的局部最小值来提高模型的泛化能力。
    它执行两步梯度更新：首先寻找使损失最大化的扰动，然后在该点进行梯度更新。

    Attributes:
        base_optimizer (torch.optim.Optimizer): 基础优化器实例（如 SGD 或 Adam）。
        param_groups (list): 包含优化器参数组的列表。
        state (dict): 存储参数状态（如扰动向量 e_w）。
    '''

    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        '''初始化 SAM 优化器。

        Args:
            params (iterable): 可迭代的模型参数或定义参数组的字典。
            base_optimizer (class): 基础优化器类（注意是类名，如 torch.optim.SGD）。
            rho (float, optional): 邻域大小，用于控制扰动范围。默认为 0.05。
            **kwargs: 传递给基础优化器的其他超参数（如 lr, momentum, weight_decay）。

        Raises:
            AssertionError: 如果 rho 小于 0。
        '''
        assert rho >= 0.0, f'无效的 rho 值: {rho}，必须是非负数。'

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        r'''计算并应用参数扰动 epsilon。

        该步骤根据当前梯度将模型参数 $w$ 更新为 $w + \epsilon$。

        Args:
            zero_grad (bool, optional): 是否在计算后清除梯度。默认为 False。
        '''
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)

            for p in group['params']:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                self.state[p]['e_w'] = e_w  # 存储扰动用于第二步恢复
                p.add_(e_w)  # 参数移动到 w + e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        r'''恢复参数并执行真正的梯度更新。

        该步骤将参数从 $w + \epsilon$ 恢复回 $w$，并利用扰动位置的梯度更新 $w$。

        Args:
            zero_grad (bool, optional): 是否在更新后清除梯度。默认为 False。
        '''
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                p.sub_(self.state[p]['e_w'])  # 恢复到原始参数 w

        self.base_optimizer.step()  # 使用 w + e_w 处的梯度进行实际更新

        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        '''计算所有参数梯度的全局 L2 范数。

        Returns:
            torch.Tensor: 梯度的 L2 范数标量。
        '''
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group['params']
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def step(self, closure=None):
        raise NotImplementedError('SAM requires steps to be run manually: first_step and second_step')

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups

    def state_dict(self):
        return self.base_optimizer.state_dict()
