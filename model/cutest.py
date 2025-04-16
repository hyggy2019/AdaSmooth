import torch
from .synthetic_functions import SyntheticFunction
from typing import Tuple
from pycutest import import_problem

class CUTESTFunction(SyntheticFunction):
    def __init__(
        self,
        name: str,
        x_init: torch.Tensor
    ):
        super(CUTESTFunction, self).__init__(x_init)

        self.name = name

    def forward(self) -> torch.Tensor:
        p = import_problem(self.name, sifParams={
            'N': self.dim,
        })
        x = self.x.detach().numpy()

        f, g = p.obj(x, gradient=True)

        f = torch.tensor(f, dtype=torch.float)
        g = torch.from_numpy(g).float()
        gx = torch.sum(g * self.x)

        # for cutest backward compatibility
        return gx + (f - gx).detach()
