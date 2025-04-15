from typing import Iterator, Tuple
import torch
import torch.nn as nn

# Adam
class True_Gradient(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterator[torch.Tensor],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.99),
        epsilon: float = 1e-8,
    ):
        if not lr >= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if not epsilon >= 0.0:
            raise ValueError(f"Invalid epsilon value: {epsilon}")

        defaults = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            epsilon=epsilon,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                loss.backward()

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]

                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(param.data)
                    state['v'] = torch.zeros_like(param.data)

                m, v = state['m'], state['v']
                lr = group['lr']
                beta1, beta2 = group['beta1'], group['beta2']
                epsilon = group['epsilon']

                state['step'] += 1

                m.mul_(beta1).add_((1 - beta1) * grad)
                v.mul_(beta2).add_((1 - beta2) * (grad ** 2))
                m_hat = m / (1 - beta1 ** state['step'])
                v_hat = v / (1 - beta2 ** state['step'])
                param.add_(-lr * m_hat / (v_hat.sqrt() + epsilon))

        return loss
