import torch
from typing import Iterator, Tuple
from easydict import EasyDict
import torch.nn as nn

class Finite_Difference(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterator[torch.Tensor],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.99),
        epsilon: float = 1e-8,
        num_queries: int = 10,
        mu: float = 0.01,
    ):
        if not lr >= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if not epsilon >= 0.0:
            raise ValueError(f"Invalid epsilon value: {epsilon}")
        if not num_queries > 0:
            raise ValueError(f"Invalid number of queries: {num_queries}")
        if not mu >= 0.0:
            raise ValueError(f"Invalid mu value: {mu}")


        self.num_queries = num_queries
        self.mu = mu

        defaults = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            epsilon=epsilon
        )

        super().__init__(params, defaults)

    def _generate_random_direction(self):
        random_direction = []
        for group in self.param_groups:
            random_direction.append([])

            for param in group['params']:
                random_direction[-1].append(torch.randn_like(param))

        return random_direction

    def _perturb_params(self, random_direction, mu):
        for i, group in enumerate(self.param_groups):
            for param, perturbation in zip(group['params'], random_direction[i]):
                param.add_(mu * perturbation)

    @torch.no_grad()
    def step(self, closure):
        assert closure is not None, "Closure function is required for finite difference optimization"

        loss = closure()

        ds = []
        fs = []
        for _ in range(self.num_queries):
            random_direction = self._generate_random_direction()
            ds.append(random_direction)
            self._perturb_params(random_direction, self.mu)
            f_x_plus_h = closure()
            fs.append(f_x_plus_h.item())
            self._perturb_params(random_direction, -self.mu)

        fs = torch.tensor(fs, device=loss.device)
        # fs_mean = torch.mean(fs) # a large mu should use this one, while a smaller one should use the following one
        fs_mean = loss

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    param.grad = torch.empty_like(param)
                param.grad.zero_()
        
        for d, f_x_plus_h in zip(ds, fs):
            for i, group in enumerate(self.param_groups):
                for params, perturbation in zip(group['params'], d[i]):
                    params.grad += (f_x_plus_h - fs_mean) / self.mu * perturbation

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad
                grad.div_(self.num_queries) # ZO algorithm divide by the number of queries
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

                m.mul_(beta1).add_((1 - beta1) * param.grad)
                v.mul_(beta2).add_((1 - beta2) * (param.grad ** 2))
                m_hat = m / (1 - beta1 ** state['step'])
                v_hat = v / (1 - beta2 ** state['step'])
                param.add_(-lr * m_hat / (v_hat.sqrt() + epsilon))

        return loss

