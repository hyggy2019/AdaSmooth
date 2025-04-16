import torch
from typing import Iterator, Tuple

class Reinforcement_Learning_w_History(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterator[torch.Tensor],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.99),
        epsilon: float = 1e-8,
        num_queries: int = 10,
        mu: float = 0.01,
        num_histories: int = 15,
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
        self.num_histories = num_histories
        self.past = []

        defaults = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            epsilon=epsilon
        )

        super().__init__(params, defaults)

    def _generate_noise(self):
        noise = []
        for group in self.param_groups:
            noise.append([])

            for param in group['params']:
                noise[-1].append(torch.randn_like(param))

        return noise

    def _perturb_params(self, noise, mu):
        for i, group in enumerate(self.param_groups):
            for param, perturbation in zip(group['params'], noise[i]):
                param.add_(mu * perturbation)

    @torch.no_grad()
    def step(self, closure):
        assert closure is not None, "Closure function is required for reinforcement learning with history optimization"

        loss = closure()

        for _ in range(self.num_queries):
            noise = self._generate_noise()
            self._perturb_params(noise, self.mu)
            reward = closure()
            self._perturb_params(noise, -self.mu)

            self.past.append([noise, reward])

        if len(self.past) > self.num_histories * self.num_queries:
            self.past = self.past[-self.num_histories * self.num_queries:]
        
        noises = [p[0] for p in self.past]
        rewards = [p[1] for p in self.past]
        rewards = torch.tensor(rewards, device=loss.device)
        rewards = rewards - rewards.mean()
        
        for noise, reward in zip(noises, rewards):
            for i, group in enumerate(self.param_groups):
                for param, perturbation in zip(group['params'], noise[i]):
                    if param.grad is None:
                        param.grad = torch.empty_like(param)

                    param.grad += reward / self.mu * perturbation

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad
                grad.div_(len(rewards)) # rl w/ history algorithm divide by the number of rewards
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