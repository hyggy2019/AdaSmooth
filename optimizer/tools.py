import math
from typing import Iterator
from easydict import EasyDict
import torch
from .fo import True_Gradient
from .zo import Finite_Difference
from .relizo import LIZO

def get_optimizer(
    name: str,
    params: Iterator[torch.Tensor],
    args: EasyDict
) -> torch.optim.Optimizer:
    """
    Get the optimizer class based on the name.
    """
    
    if name == "fo":
        return True_Gradient(params=params, lr=args.lr, betas=args.betas, epsilon=args.epsilon)
    elif name == "zo":
        return Finite_Difference(params=params, lr=args.lr, betas=args.betas, epsilon=args.epsilon, num_queries=args.num_queries, mu=math.sqrt(args.variance))
    elif name == "rl":
        return True_Gradient(params=params, lr=args.lr, betas=args.betas, epsilon=args.epsilon)
    elif name == "rl_w_history":
        return True_Gradient(params=params, lr=args.lr, betas=args.betas, epsilon=args.epsilon)
    elif name == "relizo":
        return LIZO(
            params=params,
            lr=args.lr,
            weight_decay=0.0,
            num_sample_per_step=args.num_queries,
            max_reuse_rate=0.5,
            orthogonal_sample=False,
            line_search_fn=None,
        )
    else:
        raise ValueError(f"Unknown optimizer name: {name}, available optimizers are: fo, zo, reinforce, relizo")