from collections import defaultdict
import torch
from typing import Iterator, Tuple

class ZerothOrderOptimizer(torch.optim.Optimizer):
    """
    Base class for zeroth-order optimizers.
    """
    def __init__(
        self,
        params: Iterator[torch.Tensor],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.99),
        epsilon: float = 1e-8,
        num_queries: int = 10,
        mu: float = 0.01,
        update_rule: str = 'radazo',
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
        self.update_rule = update_rule

        defaults = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            epsilon=epsilon
        )

        super().__init__(params, defaults)

    def _generate_noise(self):
        noise = {}
        for group in self.param_groups:
            for param in group['params']:
                noise[param] = torch.randn_like(param)

        return noise

    def _perturb_params(self, noise, mu):
        for group in self.param_groups:
            for param in group['params']:
                param.add_(mu * noise[param])

    def estimate_gradient(self, closure):
        """
        Estimate the gradient using finite differences.
        """
        
        raise NotImplementedError("This method should be implemented in subclasses.")

    @torch.no_grad()
    def step(self, closure):
        assert closure is not None, "Closure function is required for zeroth order optimization"

        loss = self.estimate_gradient(closure)

        if self.update_rule == 'sgd':
            for group in self.param_groups:
                for param in group['params']:
                    if param.grad is None:
                        continue
                    lr = group['lr']
                    param.add_(-lr * param.grad)
        else: # adam radazo
            for group in self.param_groups:
                for param in group['params']:
                    if param.grad is None:
                        continue

                    grad = param.grad
                    state = self.state[param]

                    if len(state) == 0:
                        state['step'] = 0
                        state['m'] = torch.zeros_like(param)
                        state['v'] = torch.zeros_like(param)

                    m, v = state['m'], state['v']
                    lr = group['lr']
                    beta1, beta2 = group['beta1'], group['beta2']
                    epsilon = group['epsilon']

                    state['step'] += 1

                    m.mul_(beta1).add_((1 - beta1) * grad)
                    if self.update_rule == 'radazo':
                        v.mul_(beta2).add_((1 - beta2) * (m ** 2))
                    else:
                        v.mul_(beta2).add_((1 - beta2) * (grad ** 2))
                    # m_hat = m / (1 - beta1 ** state['step'])
                    # v_hat = v / (1 - beta2 ** state['step'])
                    m_hat = m
                    v_hat = v
                    param.add_(-lr * m_hat / (v_hat.sqrt() + epsilon))

        return loss

class Vanilla(ZerothOrderOptimizer):
    def estimate_gradient(self, closure):
        """
        Estimate the gradient using finite differences.
        """
        loss = closure()

        noises = []
        fs = []
        for _ in range(self.num_queries):
            noise = self._generate_noise()
            noises.append(noise)
            self._perturb_params(noise, self.mu)
            f_x_plus_h = closure()
            fs.append(f_x_plus_h.item())
            self._perturb_params(noise, -self.mu)

        fs = torch.tensor(fs, device=loss.device)
        fs_mean = loss
        
        for group in self.param_groups:
            for param in group['params']:
                for noise, f_x_plus_h in zip(noises, fs):
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)
                    
                    param.grad += (f_x_plus_h - fs_mean) / self.mu * noise[param]

                param.grad.div_(self.num_queries) # ZO algorithm divide by the number of queries

        return loss

class Reinforcement_Learning(ZerothOrderOptimizer):
    def estimate_gradient(self, closure):
        loss = closure()

        noises = []
        rewards = []
        for _ in range(self.num_queries):
            noise = self._generate_noise()
            noises.append(noise)
            self._perturb_params(noise, self.mu)
            reward = closure()
            rewards.append(reward)
            self._perturb_params(noise, -self.mu)

        rewards = torch.tensor(rewards, device=loss.device)
        inds = torch.argsort(rewards)
        rewards[inds] = 2 * torch.arange(self.num_queries, device=rewards.device, dtype=rewards.dtype)
        rewards = (rewards - rewards.mean())
        
        for group in self.param_groups:
            for param in group['params']:
                for noise, reward in zip(noises, rewards):
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)

                    param.grad += reward / self.mu * noise[param]

                param.grad.div_(self.num_queries) # ZO algorithm divide by the number of queries

        return loss

class ZoAR(ZerothOrderOptimizer):
    def __init__(
        self,
        params: Iterator[torch.Tensor],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.99),
        epsilon: float = 1e-8,
        num_queries: int = 10,
        mu: float = 0.01,
        update_rule: str = 'radazo',
        num_histories: int = 5,
    ):
        super().__init__(params, lr, betas, epsilon, num_queries, mu, update_rule)

        self.num_histories = num_histories
        self.past = []

    def estimate_gradient(self, closure):
        loss = closure()

        for _ in range(self.num_queries):
            noise = self._generate_noise()
            self._perturb_params(noise, self.mu)
            reward = closure()
            self._perturb_params(noise, -self.mu)

            self.past.append([noise, reward])

        if len(self.past) > (self.num_histories + 1) * self.num_queries:
            self.past = self.past[-(self.num_histories + 1) * self.num_queries:]
        
        noises = [p[0] for p in self.past]
        rewards = [p[1] for p in self.past]
        rewards = torch.tensor(rewards, device=loss.device)
        rewards = rewards - rewards.mean()
        
        for group in self.param_groups:
            for param in group['params']:
                for noise, reward in zip(noises, rewards):
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)

                    param.grad += reward / self.mu * noise[param]

                param.grad.div_(len(rewards)) # rl w/ history algorithm divide by the number of rewards

        return loss
    
class ZoHS(ZerothOrderOptimizer):
    def __init__(
        self,
        params: Iterator[torch.Tensor],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.99),
        epsilon: float = 1e-8,
        num_queries: int = 10,
        mu: float = 0.01,
        update_rule: str = 'radazo',
        num_histories: int = 5,
    ):
        super().__init__(params, lr, betas, epsilon, num_queries, mu, update_rule)

        self.num_histories = num_histories
        self.past = []

    def estimate_gradient(self, closure):
        loss = closure()

        noises = []
        fs = []
        for _ in range(self.num_queries):
            noise = self._generate_noise()
            noises.append(noise)
            self._perturb_params(noise, self.mu)
            f_x_plus_h = closure()
            fs.append(f_x_plus_h.item())
            self._perturb_params(noise, -self.mu)
        
        fs = torch.tensor(fs, device=loss.device)
        fs_mean = loss

        this_step_grads = defaultdict(lambda: None)
        for group in self.param_groups:
            for param in group['params']:
                for noise, f_x_plus_h in zip(noises, fs):
                    if this_step_grads[param] is None:
                        this_step_grads[param] = torch.zeros_like(param)
                    
                    this_step_grads[param] += (f_x_plus_h - fs_mean) / self.mu * noise[param]

                this_step_grads[param].div_(self.num_queries) # ZO algorithm divide by the number of queries
        
        self.past.append(this_step_grads)

        if len(self.past) > self.num_histories + 1:
            self.past = self.past[-(self.num_histories + 1):]
        
        for group in self.param_groups:
            for param in group['params']:
                history_grad = [p[param] for p in self.past]

                param.grad = torch.mean(torch.stack(history_grad), dim=0)

        return loss
    
class ZoHS_Expavg(ZerothOrderOptimizer):
    def __init__(
        self,
        params: Iterator[torch.Tensor],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.99),
        epsilon: float = 1e-8,
        num_queries: int = 10,
        mu: float = 0.01,
        update_rule: str = 'radazo',
    ):
        super().__init__(params, lr, betas, epsilon, num_queries, mu, update_rule)

        self.expavg_grads = defaultdict(lambda: None)

    def estimate_gradient(self, closure):
        loss = closure()

        noises = []
        fs = []
        for _ in range(self.num_queries):
            noise = self._generate_noise()
            noises.append(noise)
            self._perturb_params(noise, self.mu)
            f_x_plus_h = closure()
            fs.append(f_x_plus_h.item())
            self._perturb_params(noise, -self.mu)
        
        fs = torch.tensor(fs, device=loss.device)
        fs_mean = loss

        beta3 = 0.9
        for group in self.param_groups:
            for param in group['params']:
                grad = torch.zeros_like(param)

                for noise, f_x_plus_h in zip(noises, fs):
                    grad += (f_x_plus_h - fs_mean) / self.mu * noise[param]

                grad.div_(self.num_queries) # ZO algorithm divide by the number of queries

                if self.expavg_grads[param] is None:
                    self.expavg_grads[param] = grad
                else:
                    self.expavg_grads[param].mul_(beta3).add_((1 - beta3) * grad)

                param.grad = self.expavg_grads[param].clone()

        return loss
    

class ZOO(ZerothOrderOptimizer):
    def __init__(
        self,
        params: Iterator[torch.Tensor],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.99),
        epsilon: float = 1e-8,
        num_queries: int = 10,
        mu: float = 0.01,
        update_rule: str = 'radazo',
        baseline: str = "single",
    ):
        super().__init__(params, lr, betas, epsilon, num_queries, mu, update_rule)

        self.baseline = baseline

    def estimate_gradient(self, closure):
        """
        Estimate the gradient using finite differences.
        """
        loss = closure()

        noises = []
        fs = []
        for _ in range(self.num_queries):
            noise = self._generate_noise()
            noises.append(noise)
            self._perturb_params(noise, self.mu)
            f_x_plus_h = closure()
            fs.append(f_x_plus_h.item())
            self._perturb_params(noise, -self.mu)

        fs = torch.tensor(fs, device=loss.device)
        if self.baseline == "single":
            fs_mean = loss
        else: # "average"
            fs_mean = torch.mean(fs)
        
        for group in self.param_groups:
            for param in group['params']:
                for noise, f_x_plus_h in zip(noises, fs):
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)
                    
                    param.grad += (f_x_plus_h - fs_mean) / self.mu * noise[param]

                param.grad.div_(self.num_queries) # ZO algorithm divide by the number of queries

        return loss
    
class REINFORCE(ZerothOrderOptimizer):
    def __init__(
        self,
        params: Iterator[torch.Tensor],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.99),
        epsilon: float = 1e-8,
        num_queries: int = 10,
        mu: float = 0.01,
        update_rule: str = 'radazo',
        baseline: str = "single",
    ):
        super().__init__(params, lr, betas, epsilon, num_queries, mu, update_rule)

        self.baseline = baseline

    def estimate_gradient(self, closure):
        loss = closure()

        rewards = []
        noises = []
        for _ in range(self.num_queries):
            noise = self._generate_noise()
            self._perturb_params(noise, self.mu)
            reward = closure()
            self._perturb_params(noise, -self.mu)

            rewards.append(reward)
            noises.append(noise)
        
        rewards = torch.tensor(rewards, device=loss.device)
        if self.baseline == "single":
            rewards = rewards - loss
        else: # "average"
            rewards = rewards - rewards.mean()
        
        for group in self.param_groups:
            for param in group['params']:
                for noise, reward in zip(noises, rewards):
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)

                    param.grad += reward / self.mu * noise[param]

                param.grad.div_(len(rewards))

        return loss