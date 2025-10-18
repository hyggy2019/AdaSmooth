import random
import numpy as np
import torch
from typing import Iterator
from easydict import EasyDict
import torch
from optimizer.fo import True_Gradient
from optimizer.zo import (
    Vanilla, 
    Reinforcement_Learning, 
    ZoAR, 
    ZoHS, 
    ZoHS_Expavg,
    ZOO,
    REINFORCE,
)
from optimizer.relizo_adam import LIZO, _backtracking

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_optimizer(
    name: str,
    params: Iterator[torch.Tensor],
    args: EasyDict
) -> torch.optim.Optimizer:
    """
    Get the optimizer class based on the name.
    """
    
    if name == "fo":
        return True_Gradient(params=params, lr=args.lr, betas=args.betas, epsilon=args.epsilon, update_rule=args.update_rule)
    elif name == "vanilla":
        return Vanilla(params=params, lr=args.lr, betas=args.betas, epsilon=args.epsilon, num_queries=args.num_queries, mu=args.mu, update_rule=args.update_rule)
    elif name == "rl":
        return Reinforcement_Learning(params=params, lr=args.lr, betas=args.betas, epsilon=args.epsilon, num_queries=args.num_queries, mu=args.mu, update_rule=args.update_rule)
    elif name == "zoar":
        return ZoAR(params=params, lr=args.lr, betas=args.betas, epsilon=args.epsilon, num_queries=args.num_queries, mu=args.mu, num_histories=args.num_histories, update_rule=args.update_rule)
    elif name == "zoar_0":
        return ZoAR(params=params, lr=args.lr, betas=args.betas, epsilon=args.epsilon, num_queries=args.num_queries, mu=args.mu, num_histories=0, update_rule=args.update_rule)
    elif name == "relizo":
        return LIZO(
            params=params,
            lr=args.lr,
            weight_decay=0.0,
            num_sample_per_step=args.num_queries,
            reuse_distance_bound=2*args.lr,
            max_reuse_rate=0.5,
            orthogonal_sample=False,
            fast_alg=False,
            # line_search_fn=partial(_backtracking),
            line_search_fn=None,
            # strict_lr=True,
            sample_norm=args.mu,
            betas=args.betas,
            eps=args.epsilon,
            update_rule=args.update_rule,
        )
    elif name == "zohs":
        return ZoHS(params=params, lr=args.lr, betas=args.betas, epsilon=args.epsilon, num_queries=args.num_queries, mu=args.mu, num_histories=args.num_histories, update_rule=args.update_rule)
    elif name == "zohs_expavg":
        return ZoHS_Expavg(params=params, lr=args.lr, betas=args.betas, epsilon=args.epsilon, num_queries=args.num_queries, mu=args.mu, update_rule=args.update_rule)
    elif name == "zoo":
        return ZOO(params=params, lr=args.lr, betas=args.betas, epsilon=args.epsilon, num_queries=args.num_queries, mu=args.mu, update_rule=args.update_rule, baseline=args.baseline)
    elif name == "reinforce":
        return REINFORCE(params=params, lr=args.lr, betas=args.betas, epsilon=args.epsilon, num_queries=args.num_queries, mu=args.mu, update_rule=args.update_rule, baseline=args.baseline)
    else:
        raise ValueError(f"Unknown optimizer name: {name}, available optimizers are: fo, vanilla, rl, zoar, zohs, zohs_expavg, relizo, zoo, reinforce")