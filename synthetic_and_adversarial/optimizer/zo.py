from collections import defaultdict
import torch
from typing import Iterator, Tuple
import numpy as np
from cmaes import SepCMA

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

class xNES(ZerothOrderOptimizer):
    """
    Exponential Natural Evolution Strategies (xNES).

    Adapts the search distribution by maintaining and updating:
    - mu: mean (search center)
    - sigma: step size (global scaling)
    - B: covariance matrix shape

    Reference: "Natural Evolution Strategies" (https://arxiv.org/abs/1106.4487)
    """
    def __init__(
        self,
        params: Iterator[torch.Tensor],
        lr: float = 1.0,
        betas: Tuple[float, float] = (0.9, 0.99),
        epsilon: float = 1e-8,
        num_queries: int = 10,
        mu: float = 0.01,
        update_rule: str = 'sgd',
        eta_mu: float = 1.0,
        eta_sigma: float = None,
        eta_bmat: float = None,
        use_fshape: bool = True,
        initial_sigma: float = 0.1,
    ):
        # Validate that xNES uses SGD
        if update_rule != 'sgd':
            raise ValueError("xNES requires update_rule='sgd'")

        super().__init__(params, lr, betas, epsilon, num_queries, mu, update_rule)

        self.eta_mu = eta_mu
        self.use_fshape = use_fshape
        self.initial_sigma = initial_sigma

        # Initialize xNES-specific state
        self.initialized = False
        self.dim = None
        self.sigma_xnes = None
        self.bmat = None
        self.eta_sigma = eta_sigma
        self.eta_bmat = eta_bmat
        self.utilities = None

    def _initialize_xnes(self, param):
        """Initialize xNES parameters on first call"""
        if self.initialized:
            return

        self.dim = param.numel()
        self.sigma_xnes = self.initial_sigma  # Use initial_sigma for proper step size
        self.bmat = torch.eye(self.dim, device=param.device, dtype=param.dtype)

        # Set default learning rates if not specified
        dim = self.dim
        if self.eta_sigma is None:
            self.eta_sigma = 3 * (3 + torch.log(torch.tensor(dim, dtype=torch.float32))) / (5 * dim * torch.sqrt(torch.tensor(dim, dtype=torch.float32)))
        if self.eta_bmat is None:
            self.eta_bmat = 3 * (3 + torch.log(torch.tensor(dim, dtype=torch.float32))) / (5 * dim * torch.sqrt(torch.tensor(dim, dtype=torch.float32)))

        # Compute utilities for fitness shaping
        if self.use_fshape:
            npop = self.num_queries
            a = torch.log(torch.tensor(1 + 0.5 * npop))
            utilities = torch.tensor([max(0, a - torch.log(torch.tensor(k, dtype=torch.float32))) for k in range(1, npop + 1)])
            utilities /= utilities.sum()
            utilities -= 1.0 / npop
            utilities = torch.flip(utilities, [0])  # Ascending order
            self.utilities = utilities.to(param.device)

        self.initialized = True

    def estimate_gradient(self, closure):
        """xNES gradient estimation with adaptive covariance"""
        # Get the parameter (assume single parameter for now)
        param = None
        for group in self.param_groups:
            for p in group['params']:
                param = p
                break
            if param is not None:
                break

        self._initialize_xnes(param)

        # Save original parameter
        param_original = param.data.clone()
        param_flat = param_original.view(-1)

        # Sample perturbations
        s_samples = []
        f_samples = []

        for _ in range(self.num_queries):
            # Sample from standard normal
            s = torch.randn(self.dim, device=param.device, dtype=param.dtype)
            s_samples.append(s)

            # Transform: z = mu + sigma * B @ s
            z = param_flat + self.sigma_xnes * torch.matmul(self.bmat, s)

            # Evaluate
            param.data = z.view_as(param)
            f_val = closure()
            f_samples.append(f_val.item())

        # Restore parameter
        param.data = param_original

        # Sort by fitness
        f_samples = torch.tensor(f_samples, device=param.device)
        sorted_indices = torch.argsort(f_samples)
        f_samples = f_samples[sorted_indices]
        s_samples = torch.stack(s_samples)[sorted_indices]

        # Use utilities or raw fitness
        if self.use_fshape:
            u_samples = self.utilities
        else:
            u_samples = f_samples

        # Compute natural gradients
        dj_delta = torch.sum(u_samples.unsqueeze(1) * s_samples, dim=0)

        # Compute M matrix gradient
        eyemat = torch.eye(self.dim, device=param.device, dtype=param.dtype)
        dj_mmat = torch.matmul(s_samples.T, s_samples * u_samples.unsqueeze(1)) - u_samples.sum() * eyemat
        dj_sigma = torch.trace(dj_mmat) / self.dim
        dj_bmat = dj_mmat - dj_sigma * eyemat

        # Update xNES parameters
        # mu update: converted to gradient for standard optimizer
        grad_direction = torch.matmul(self.bmat, dj_delta)

        # Update sigma and B matrix
        self.sigma_xnes = self.sigma_xnes * torch.exp(0.5 * self.eta_sigma * dj_sigma)
        self.bmat = torch.matmul(self.bmat, torch.matrix_exp(0.5 * self.eta_bmat * dj_bmat))

        # Set gradient for parameter update
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                # Use xNES natural gradient scaled by eta_mu and sigma
                p.grad = -(self.eta_mu * self.sigma_xnes * grad_direction).view_as(p)

        # Return mean fitness
        return f_samples.mean()

class ES(ZerothOrderOptimizer):
    """
    Pure Evolution Strategies (ES) - Original formulation from paper Algorithm 1.

    Gradient estimator without baseline subtraction:
        ∇f(θ) ≈ (1/nσ) Σ F(θ + σεi) · εi

    This is the theoretical ES form with highest variance but no bias.
    """
    def estimate_gradient(self, closure):
        # Note: No baseline evaluation needed for pure ES
        noises = []
        fs = []

        for _ in range(self.num_queries):
            noise = self._generate_noise()
            noises.append(noise)
            self._perturb_params(noise, self.mu)
            f_perturbed = closure()
            fs.append(f_perturbed.item())
            self._perturb_params(noise, -self.mu)

        fs = torch.tensor(fs, device=next(iter(self.param_groups[0]['params'])).device)

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)

                for noise, f_val in zip(noises, fs):
                    # Pure ES: use f_val directly without baseline subtraction
                    param.grad += f_val / self.mu * noise[param]

                param.grad.div_(self.num_queries)

        # Return the mean of sampled values as loss estimate
        return fs.mean()

class TwoPointMatched(ZerothOrderOptimizer):
    """
    Two-point gradient estimator with matched query budget.
    Uses num_queries//2 directions to match one-point's total queries.

    Total queries: 1 (baseline) + num_queries (same as Vanilla).
    """
    def estimate_gradient(self, closure):
        loss = closure()  # baseline f(θ)

        num_directions = self.num_queries // 2  # Each direction uses 2 queries

        noises = []
        fs_plus = []
        fs_minus = []

        for _ in range(num_directions):
            noise = self._generate_noise()
            noises.append(noise)

            # f(θ + μu)
            self._perturb_params(noise, self.mu)
            f_plus = closure()
            fs_plus.append(f_plus.item())
            self._perturb_params(noise, -self.mu)

            # f(θ - μu)
            self._perturb_params(noise, -self.mu)
            f_minus = closure()
            fs_minus.append(f_minus.item())
            self._perturb_params(noise, self.mu)

        fs_plus = torch.tensor(fs_plus, device=loss.device)
        fs_minus = torch.tensor(fs_minus, device=loss.device)

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)

                for noise, f_p, f_m in zip(noises, fs_plus, fs_minus):
                    param.grad += (f_p - f_m) / (2 * self.mu) * noise[param]

                # Divide by number of directions (not number of queries)
                param.grad.div_(num_directions)

        return loss

class SepCMAES(torch.optim.Optimizer):
    """
    Separable CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

    Sep-CMA-ES limits the covariance matrix to a diagonal form, making it
    well-suited for high-dimensional optimization tasks.

    This implementation wraps the `cmaes` library's SepCMA class.

    Note: This optimizer does NOT use gradients and operates independently
    from the ZerothOrderOptimizer framework.

    Reference: Ros and Hansen (2008)
    """
    def __init__(
        self,
        params: Iterator[torch.Tensor],
        lr: float = 0.001,  # Not used by CMA-ES, kept for compatibility
        sigma: float = 0.1,
        population_size: int = None,
        **kwargs
    ):
        """
        Args:
            params: Iterator of parameters to optimize
            lr: Learning rate (not used by CMA-ES, kept for interface compatibility)
            sigma: Initial step size for CMA-ES
            population_size: Population size (if None, uses library default)
        """
        defaults = dict(lr=lr, sigma=sigma, population_size=population_size)
        super().__init__(params, defaults)

        # Flatten all parameters into a single vector
        self.param_shapes = []
        self.param_numel = []
        all_params = []

        for group in self.param_groups:
            for param in group['params']:
                self.param_shapes.append(param.shape)
                self.param_numel.append(param.numel())
                all_params.append(param.detach().cpu().numpy().flatten())

        # Concatenate all parameters
        self.initial_params = np.concatenate(all_params)
        self.dim = len(self.initial_params)

        # Initialize SepCMA optimizer
        mean = self.initial_params.copy()
        cma_kwargs = {}
        if population_size is not None:
            cma_kwargs['population_size'] = population_size

        self.cma_optimizer = SepCMA(mean=mean, sigma=sigma, **cma_kwargs)
        self.generation = 0

    def _params_to_numpy(self) -> np.ndarray:
        """Convert PyTorch parameters to numpy vector."""
        all_params = []
        for group in self.param_groups:
            for param in group['params']:
                all_params.append(param.detach().cpu().numpy().flatten())
        return np.concatenate(all_params)

    def _numpy_to_params(self, x: np.ndarray):
        """Convert numpy vector back to PyTorch parameters."""
        offset = 0
        for group in self.param_groups:
            for param, shape, numel in zip(group['params'], self.param_shapes, self.param_numel):
                param_data = x[offset:offset+numel].reshape(shape)
                param.data = torch.from_numpy(param_data).to(param.device, param.dtype)
                offset += numel

    @torch.no_grad()
    def step(self, closure):
        """
        Perform a single optimization step.

        Args:
            closure: A closure that evaluates the model and returns the loss.

        Returns:
            The loss value from the best solution in this generation.
        """
        assert closure is not None, "Closure function is required for CMA-ES optimization"

        # Generate solutions for this generation
        solutions = []
        for _ in range(self.cma_optimizer.population_size):
            # Ask CMA-ES for a candidate solution
            x = self.cma_optimizer.ask()

            # Set parameters to this candidate
            self._numpy_to_params(x)

            # Evaluate the loss
            loss = closure()
            if isinstance(loss, torch.Tensor):
                loss = loss.item()

            # CMA-ES minimizes, so we use the loss directly
            solutions.append((x, loss))

        # Tell CMA-ES the results
        self.cma_optimizer.tell(solutions)

        # Set parameters to the current mean of the distribution
        self._numpy_to_params(self.cma_optimizer._mean)

        self.generation += 1

        # Return the best loss from this generation
        best_loss = min(loss for _, loss in solutions)
        return best_loss

    def should_stop(self):
        """Check if optimization should stop."""
        return self.cma_optimizer.should_stop()

class AdaSmoothZO(ZerothOrderOptimizer):
    """
    AdaSmooth-ZO: Zeroth-Order Optimization with Adaptive Low-Rank Sampling.

    Learns both the search mean (θ) and low-rank covariance (LL^T) through
    KL-regularized policy optimization with moment matching.

    Key features:
    - Uses low-rank sampling: x = θ + L·u (u ~ N(0, I_K))
    - Updates θ via weighted averaging (not gradient descent)
    - Maintains adaptive covariance matrix L ∈ R^(d×K)
    - Complexity: O(Kd) space and time per iteration

    Reference: "AdaSmooth-ZO: Adaptive Smoothing with Low-Rank Updates"
    """

    def __init__(
        self,
        params: Iterator[torch.Tensor],
        lr: float = 1.0,
        betas: Tuple[float, float] = (0.9, 0.99),
        epsilon: float = 1e-8,
        num_queries: int = 10,
        mu: float = 0.1,
        update_rule: str = 'sgd',
        beta_init: float = 1.0,
        beta_decay: float = 0.05,
        beta_schedule: str = 'polynomial',
    ):
        """
        Args:
            params: Iterator of parameters to optimize
            lr: Learning rate (must be 1.0 for correct updates)
            num_queries: Batch size K (rank of covariance matrix)
            mu: Initial smoothing scale for L_0
            beta_init: Initial temperature β_0
            beta_decay: Decay rate for temperature
            beta_schedule: 'constant', 'exponential', or 'polynomial'
        """
        # Force SGD update rule and lr=1.0
        if update_rule != 'sgd':
            raise ValueError("AdaSmoothZO requires update_rule='sgd'")
        if abs(lr - 1.0) > 1e-6:
            raise ValueError("AdaSmoothZO requires lr=1.0 for correct updates")

        super().__init__(
            params, lr, betas, epsilon, num_queries, mu, update_rule
        )

        self.K = num_queries
        self.initial_sigma = mu
        self.beta_init = beta_init
        self.beta_decay = beta_decay
        self.beta_schedule_type = beta_schedule
        self.iteration = 0

        # Initialize low-rank smoothing matrix L
        self._initialize_L()

        # History tracking
        self.history = {
            'f_values': [],
            'weights': [],
            'beta': [],
            'L_norms': []
        }

    def _get_dim(self):
        """Get total parameter dimension"""
        for group in self.param_groups:
            for param in group['params']:
                return param.numel()
        return 1000  # Default fallback

    def _initialize_L(self):
        """Initialize smoothing matrix L_0

        Always use diagonal L (stored as d-dim vector) for efficiency
        Similar to SepCMAES diagonal covariance
        """
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                d = param.numel()

                # Always use diagonal L: store as vector of length d
                state['L'] = self.initial_sigma * torch.ones(
                    d, device=param.device, dtype=param.dtype
                )
                state['L_is_diagonal'] = True

    def _initialize_evolution_path(self):
        """Initialize evolution path pc (like SepCMA) for cumulative gradient tracking"""
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                d = param.numel()
                state['pc'] = torch.zeros(d, device=param.device, dtype=param.dtype)

    def _get_beta(self) -> float:
        """Get current temperature β_t"""
        t = self.iteration

        if self.beta_schedule_type == 'constant':
            return self.beta_init
        elif self.beta_schedule_type == 'exponential':
            import math
            return self.beta_init * math.exp(-self.beta_decay * t)
        elif self.beta_schedule_type == 'polynomial':
            return self.beta_init / (1 + self.beta_decay * t)
        else:
            raise ValueError(f"Unknown schedule: {self.beta_schedule_type}")

    def estimate_gradient(self, closure):
        """
        AdaSmooth-ZO gradient estimation with adaptive low-rank covariance.

        Process:
        1. Sample from low-rank Gaussian: x = θ + L·u
        2. Compute exponential weights: w ∝ exp(-f(x)/β)
        3. Update θ via weighted averaging
        4. Update L via weighted residuals
        5. Set param.grad to mimic SGD update: grad = (θ_old - θ_new) / lr
        """
        beta_t = self._get_beta()

        # Get parameter (assume single parameter for now)
        param = None
        for group in self.param_groups:
            for p in group['params']:
                param = p
                break
            if param is not None:
                break

        if param is None:
            raise ValueError("No parameters to optimize")

        state = self.state[param]
        L_t = state['L']
        is_diagonal = state.get('L_is_diagonal', False)

        # Flatten current parameter
        theta_t = param.data.view(-1)  # Shape: (d,)
        d = theta_t.shape[0]

        # ===== 1. Sampling (diagonal or low-rank) =====
        X = []  # Candidate solutions
        Y = []  # Function values

        if is_diagonal:
            # Diagonal L: Sample directly in parameter space
            # L is (d,) vector, x = θ + L ⊙ u where u ~ N(0, I_d)
            for k in range(self.K):
                u = torch.randn(d, device=param.device, dtype=param.dtype)
                x_k = theta_t + L_t * u  # Element-wise multiplication
                # Set parameter and evaluate
                param.data = x_k.view_as(param)
                f_val = closure()
                if isinstance(f_val, torch.Tensor):
                    f_val = f_val.item()
                X.append(x_k)
                Y.append(f_val)
        else:
            # Low-rank L: Sample in latent space
            # L is (d, K) matrix, x = θ + L @ u where u ~ N(0, I_K)
            K = L_t.shape[1]
            for k in range(K):
                u = torch.randn(K, device=param.device, dtype=param.dtype)
                x_k = theta_t + torch.matmul(L_t, u)

                # Set parameter and evaluate
                param.data = x_k.view_as(param)
                f_val = closure()

                if isinstance(f_val, torch.Tensor):
                    f_val = f_val.item()

                X.append(x_k)
                Y.append(f_val)

        # Stack to tensors
        X = torch.stack(X)  # Shape: (K, d)
        Y = torch.tensor(Y, device=param.device, dtype=param.dtype)

        # ===== 2. Compute Weights (KL-divergence with baseline) =====
        # KEY FIX: Subtract baseline (mean) to get advantages
        # This prevents weight concentration when function values are large
        baseline = Y.mean()  # Baseline: average function value
        advantages = Y - baseline  # Advantages: relative performance

        # Exponential weighting with numerical stability
        # Use log-sum-exp trick to avoid overflow/underflow
        log_weights = -advantages / beta_t
        log_weights = log_weights - log_weights.max()  # Numerical stability
        V = torch.exp(log_weights)

        # Normalize: w_k = v_k / Σv_j
        W = V / V.sum()

        # Handle edge case: if all weights are zero
        if torch.isnan(W).any() or torch.isinf(W).any():
            W = torch.ones_like(W) / len(W)  # Uniform weights

        # ===== 3. Update Mean (via weighted averaging) =====
        # θ_{t+1} = Σ w_k · x_k
        theta_new = torch.sum(W.unsqueeze(1) * X, dim=0)  # Shape: (d,)

        # ===== 4. Update Smoothing Matrix =====
        # Compute residuals: x_k - θ_{t+1}
        residuals = X - theta_new.unsqueeze(0)  # Shape: (K, d)

        # Weighted residuals: c_k = √w_k · (x_k - θ_{t+1})
        weighted_residuals = torch.sqrt(W).unsqueeze(1) * residuals  # (K, d)

        if is_diagonal:
            # Diagonal L: Compute element-wise standard deviation
            # L_new[i] = sqrt(Σ_k w_k · (x_k[i] - θ_new[i])^2)
            # This is equivalent to the diagonal of sqrt(LL^T)
            weighted_sq_residuals = W.unsqueeze(1) * (residuals ** 2)  # (K, d)
            L_new = torch.sqrt(weighted_sq_residuals.sum(dim=0) + 1e-8)  # (d,)
        else:
            # Low-rank L: Stack weighted residuals as columns (original AdaSmooth)
            # L_{t+1} = [c_1, c_2, ..., c_K] ∈ R^(d × K)
            L_new = weighted_residuals.T  # Shape: (d, K)

        # Update L in state
        state['L'] = L_new

        # ===== 6. Set pseudo-gradient to achieve θ_new via SGD =====
        # We want: θ ← θ - lr·grad = θ_new
        # So: grad = (θ - θ_new) / lr
        lr = self.param_groups[0]['lr']
        pseudo_grad = (theta_t - theta_new) / lr

        # Set gradient
        param.grad = pseudo_grad.view_as(param)

        # ===== 7. Track History =====
        self.history['f_values'].append(Y.cpu().numpy())
        self.history['weights'].append(W.cpu().numpy())
        self.history['beta'].append(beta_t)
        self.history['L_norms'].append(torch.norm(L_new).item())

        self.iteration += 1

        # Return weighted mean loss
        weighted_loss = torch.sum(W * Y)
        return weighted_loss

class AdaSmoothZO_MultiParam(ZerothOrderOptimizer):
    """
    AdaSmooth-ZO for models with multiple parameter tensors.

    Flattens all parameters into a single vector for unified covariance learning.
    This allows the algorithm to capture correlations across different parameter tensors.
    """

    def __init__(
        self,
        params: Iterator[torch.Tensor],
        lr: float = 1.0,
        betas: Tuple[float, float] = (0.9, 0.99),
        epsilon: float = 1e-8,
        num_queries: int = 64,
        mu: float = 0.1,
        update_rule: str = 'sgd',
        beta_init: float = 1.0,
        beta_decay: float = 0.05,
        beta_schedule: str = 'polynomial',
    ):
        """
        Args:
            params: Iterator of parameters to optimize
            lr: Learning rate (must be 1.0)
            num_queries: Batch size K (rank of covariance matrix)
            mu: Initial smoothing scale for L_0
            beta_init: Initial temperature β_0
            beta_decay: Decay rate for temperature
            beta_schedule: Temperature schedule type
        """
        # Validate inputs
        if update_rule != 'sgd':
            raise ValueError("AdaSmoothZO_MultiParam requires update_rule='sgd'")
        if abs(lr - 1.0) > 1e-6:
            raise ValueError("AdaSmoothZO_MultiParam requires lr=1.0")

        super().__init__(
            params, lr, betas, epsilon, num_queries, mu, update_rule
        )

        self.K = num_queries
        self.initial_sigma = mu
        self.beta_init = beta_init
        self.beta_decay = beta_decay
        self.beta_schedule_type = beta_schedule
        self.iteration = 0

        # Store parameter shapes for flattening/unflattening
        self.param_shapes = []
        self.param_numels = []
        total_numel = 0

        for group in self.param_groups:
            for param in group['params']:
                self.param_shapes.append(param.shape)
                numel = param.numel()
                self.param_numels.append(numel)
                total_numel += numel

        self.total_dim = total_numel

        # Single unified L matrix for all parameters
        # Store in first parameter's state
        first_param = None
        for group in self.param_groups:
            for param in group['params']:
                first_param = param
                break
            if first_param is not None:
                break

        self.state[first_param]['L'] = self.initial_sigma * torch.randn(
            self.total_dim, self.K,
            device=first_param.device,
            dtype=first_param.dtype
        )
        self.state[first_param]['is_storage'] = True

        # History tracking
        self.history = {
            'f_values': [],
            'weights': [],
            'beta': [],
            'L_norms': []
        }

    def _get_beta(self) -> float:
        """Get current temperature β_t"""
        t = self.iteration

        if self.beta_schedule_type == 'constant':
            return self.beta_init
        elif self.beta_schedule_type == 'exponential':
            import math
            return self.beta_init * math.exp(-self.beta_decay * t)
        elif self.beta_schedule_type == 'polynomial':
            return self.beta_init / (1 + self.beta_decay * t)
        else:
            return self.beta_init / (1 + self.beta_decay * t)

    def _flatten_params(self) -> torch.Tensor:
        """Flatten all parameters into a single vector"""
        flat = []
        for group in self.param_groups:
            for param in group['params']:
                flat.append(param.data.view(-1))
        return torch.cat(flat)

    def _unflatten_to_params(self, flat: torch.Tensor):
        """Unflatten vector back to parameters"""
        offset = 0
        for group in self.param_groups:
            for param, shape, numel in zip(
                group['params'], self.param_shapes, self.param_numels
            ):
                param.data = flat[offset:offset+numel].view(shape)
                offset += numel

    def _get_L(self) -> torch.Tensor:
        """Get the unified L matrix"""
        for group in self.param_groups:
            for param in group['params']:
                if 'is_storage' in self.state[param]:
                    return self.state[param]['L']
        raise ValueError("L matrix not found")

    def _set_L(self, L_new: torch.Tensor):
        """Set the unified L matrix"""
        for group in self.param_groups:
            for param in group['params']:
                if 'is_storage' in self.state[param]:
                    self.state[param]['L'] = L_new
                    return

    def estimate_gradient(self, closure):
        """AdaSmooth-ZO gradient estimation for multi-parameter models"""
        beta_t = self._get_beta()

        # Get current flattened parameters and L
        theta_t = self._flatten_params()
        L_t = self._get_L()
        d, K = L_t.shape

        # ===== 1. Low-Rank Sampling =====
        X = []
        Y = []

        for k in range(K):
            u = torch.randn(K, device=theta_t.device, dtype=theta_t.dtype)
            x_k = theta_t + torch.matmul(L_t, u)

            # Unflatten and evaluate
            self._unflatten_to_params(x_k)
            f_val = closure()

            if isinstance(f_val, torch.Tensor):
                f_val = f_val.item()

            X.append(x_k)
            Y.append(f_val)

        X = torch.stack(X)
        Y = torch.tensor(Y, device=theta_t.device, dtype=theta_t.dtype)

        # ===== 2. Compute Weights (with numerical stability) =====
        log_weights = -Y / beta_t
        log_weights = log_weights - log_weights.max()
        V = torch.exp(log_weights)
        W = V / V.sum()

        # Handle edge case
        if torch.isnan(W).any() or torch.isinf(W).any():
            W = torch.ones_like(W) / len(W)

        # ===== 3. Update Mean =====
        theta_new = torch.sum(W.unsqueeze(1) * X, dim=0)

        # ===== 4. Update Smoothing Matrix =====
        residuals = X - theta_new.unsqueeze(0)
        weighted_residuals = torch.sqrt(W).unsqueeze(1) * residuals
        L_new = weighted_residuals.T

        self._set_L(L_new)

        # ===== 5. Set pseudo-gradients =====
        lr = self.param_groups[0]['lr']
        flat_grad = (theta_t - theta_new) / lr

        offset = 0
        for group in self.param_groups:
            for param, numel in zip(group['params'], self.param_numels):
                param.grad = flat_grad[offset:offset+numel].view_as(param)
                offset += numel

        # ===== 6. Track History =====
        self.history['f_values'].append(Y.cpu().numpy())
        self.history['weights'].append(W.cpu().numpy())
        self.history['beta'].append(beta_t)
        self.history['L_norms'].append(torch.norm(L_new).item())

        self.iteration += 1

        weighted_loss = torch.sum(W * Y)
        return weighted_loss