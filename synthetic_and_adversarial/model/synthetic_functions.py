import torch
import torch.nn as nn

class SyntheticFunction(nn.Module):
    def __init__(
        self,
        x_init: torch.Tensor
    ):
        super(SyntheticFunction, self).__init__()
        
        assert len(x_init.shape) == 1, "x_init must be a 1D tensor"

        self.dim = x_init.shape[0]
        self.x = torch.nn.Parameter(x_init)

class Levy(SyntheticFunction):

    def forward(self) -> torch.Tensor:
        x = self.x
        d = self.dim
        w = 1 + (x - 1) / 4

        term1 = torch.sin(torch.pi * w[0]) ** 2
        term2 = ((w[-1] - 1) ** 2) * (1 + torch.sin(2 * torch.pi * w[-1]) ** 2)
        term3 = torch.sum((w[:-1] - 1) ** 2 * (1 + 10 * torch.sin(torch.pi * w[:-1] + 1) ** 2))

        return term1 + term2 + term3
    
class Rosenbrock(SyntheticFunction):

    def forward(self) -> torch.Tensor:
        x = self.x

        term1 = 100 * (x[1:] - x[:-1] ** 2) ** 2
        term2 = (1 - x[:-1]) ** 2
        
        return torch.sum(term1 + term2)
    
class Ackley(SyntheticFunction):

    def forward(self) -> torch.Tensor:
        x = self.x
        a, b, c = 20, 0.2, 2 * torch.pi
        d = self.dim

        term1 = - a * torch.exp(-b * torch.sqrt(torch.sum(x ** 2) / d))
        term2 = - torch.exp(torch.sum(torch.cos(c * x)) / d)
        return term1 + term2 + a + torch.e


class Quadratic(SyntheticFunction):

    def forward(self) -> torch.Tensor:
        return 0.5 * torch.sum(self.x ** 2)

class Rastrigin(SyntheticFunction):

    def forward(self) -> torch.Tensor:
        x = self.x
        A = 10
        d = self.dim

        return A * d + torch.sum(x ** 2 - A * torch.cos(2 * torch.pi * x))

def get_synthetic_funcs(
    name: str,
    x_init: torch.Tensor
) -> SyntheticFunction:

    all_functions = {
        "ackley": Ackley,
        "levy": Levy,
        "rosenbrock": Rosenbrock,
        "quadratic": Quadratic,
        "rastrigin": Rastrigin,
    }

    assert name in all_functions, f"Function {name} not found. Available functions: {list(all_functions.keys())}"
    
    return all_functions[name](x_init)