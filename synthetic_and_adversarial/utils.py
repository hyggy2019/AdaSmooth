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
    ES,
    xNES,
    TwoPointMatched,
    SepCMAES,
    AdaSmoothZO,
    AdaSmoothZO_MultiParam,
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
    elif name == "es":
        return ES(params=params, lr=args.lr, betas=args.betas, epsilon=args.epsilon, num_queries=args.num_queries, mu=args.mu, update_rule=args.update_rule)
    elif name == "xnes":
        # xNES uses SGD update rule internally
        # IMPORTANT: xNES requires lr=1.0 because natural gradients are already scaled
        eta_mu = getattr(args, 'eta_mu', 1.0)
        eta_sigma = getattr(args, 'eta_sigma', None)
        eta_bmat = getattr(args, 'eta_bmat', None)
        use_fshape = getattr(args, 'use_fshape', True)
        initial_sigma = getattr(args, 'initial_sigma', 0.1)
        xnes_lr = getattr(args, 'xnes_lr', 1.0)  # Use xnes_lr if specified, otherwise 1.0
        return xNES(params=params, lr=xnes_lr, betas=args.betas, epsilon=args.epsilon, num_queries=args.num_queries, mu=args.mu, update_rule='sgd', eta_mu=eta_mu, eta_sigma=eta_sigma, eta_bmat=eta_bmat, use_fshape=use_fshape, initial_sigma=initial_sigma)
    elif name == "twopoint":
        return TwoPointMatched(params=params, lr=args.lr, betas=args.betas, epsilon=args.epsilon, num_queries=args.num_queries, mu=args.mu, update_rule=args.update_rule)
    elif name == "sepcmaes":
        # SepCMAES uses its own internal update rule (not gradient-based)
        sigma = getattr(args, 'sigma', args.mu)  # Use mu as default sigma if not specified
        population_size = getattr(args, 'population_size', None)
        return SepCMAES(params=params, lr=args.lr, sigma=sigma, population_size=population_size)
    elif name == "adasmooth" or name == "adasmoothzo":
        # AdaSmoothZO for single-parameter models
        # IMPORTANT: AdaSmooth needs higher rank K for high-dimensional problems
        # Recommended: K >= sqrt(d) for d-dimensional problems
        beta_init = getattr(args, 'beta_init', 1.0)
        beta_decay = getattr(args, 'beta_decay', 0.01)  # Slower decay (0.05 -> 0.01)
        beta_schedule = getattr(args, 'beta_schedule', 'polynomial')
        adasmooth_num_queries = getattr(args, 'adasmooth_num_queries', max(args.num_queries, 32))  # Default: at least 32
        return AdaSmoothZO(
            params=params,
            lr=1.0,  # Must be 1.0
            num_queries=adasmooth_num_queries,
            mu=args.mu,
            beta_init=beta_init,
            beta_decay=beta_decay,
            beta_schedule=beta_schedule
        )
    elif name == "adasmooth_multi" or name == "adasmoothzo_multi":
        # AdaSmoothZO for multi-parameter models
        beta_init = getattr(args, 'beta_init', 1.0)
        beta_decay = getattr(args, 'beta_decay', 0.01)  # Slower decay (0.05 -> 0.01)
        beta_schedule = getattr(args, 'beta_schedule', 'polynomial')
        adasmooth_num_queries = getattr(args, 'adasmooth_num_queries', max(args.num_queries, 32))  # Default: at least 32
        return AdaSmoothZO_MultiParam(
            params=params,
            lr=1.0,  # Must be 1.0
            num_queries=adasmooth_num_queries,
            mu=args.mu,
            beta_init=beta_init,
            beta_decay=beta_decay,
            beta_schedule=beta_schedule
        )
    else:
        raise ValueError(f"Unknown optimizer name: {name}, available optimizers are: fo, vanilla, rl, zoar, zohs, zohs_expavg, relizo, zoo, reinforce, es, xnes, twopoint, sepcmaes, adasmooth, adasmooth_multi")