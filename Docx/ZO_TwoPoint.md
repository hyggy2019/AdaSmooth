```python
class TwoPointMatched(ZerothOrderOptimizer):
    """
    Two-point gradient estimator with matched query budget.
    Uses num_queries//2 directions to match one-point's total queries.
    
    Total queries: 1 (baseline) + num_queries (same as Vanilla).
    """
    def estimate_gradient(self, closure):
        loss = closure()  # baseline f(θ)

        num_directions = self.num_queries // 2  # 每个方向用2个query
        
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
                
                # 注意：这里除以方向数（不是query数）
                param.grad.div_(num_directions)
        
        return loss
```