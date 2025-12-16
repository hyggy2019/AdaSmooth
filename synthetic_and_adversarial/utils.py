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
from optimizer.adasmooth_es import AdaSmoothES
from optimizer.adasmooth_es_v2 import AdaSmoothESv2
from optimizer.adaptive_beta_schedulers import get_adaptive_beta_scheduler
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
        return xNES(params=params, lr=args.lr, betas=args.betas, epsilon=args.epsilon, num_queries=args.num_queries, mu=args.mu, update_rule='sgd', eta_mu=eta_mu, eta_sigma=eta_sigma, eta_bmat=eta_bmat, use_fshape=use_fshape, initial_sigma=initial_sigma)
    elif name == "twopoint":
        return TwoPointMatched(params=params, lr=args.lr, betas=args.betas, epsilon=args.epsilon, num_queries=args.num_queries, mu=args.mu, update_rule=args.update_rule)
    elif name == "sepcmaes":
        # SepCMAES uses its own internal update rule (not gradient-based)
        sigma = getattr(args, 'sigma', args.mu)  # Use mu as default sigma if not specified
        population_size = getattr(args, 'population_size', None)
        return SepCMAES(params=params, lr=args.lr, sigma=sigma, population_size=population_size)
    elif name == "adasmooth" or name == "adasmoothzo":
        # AdaSmoothZO for single-parameter models with SepCMA-inspired covariance updates
        # Uses evolution path (pc) + rank-one/rank-mu updates like SepCMA
        # num_queries: number of samples per iteration (like SepCMA's population_size)
        # Recommended: 24-32 for d=1000 (matching SepCMA's 4+3*log(d) formula)
        # KEY FIX: Much slower beta decay to prevent weight concentration
        beta_init = getattr(args, 'beta_init', 10.0)  # Higher initial temperature (1.0 → 10.0)
        beta_decay = getattr(args, 'beta_decay', 0.001)  # Much slower decay (0.05 → 0.001)
        beta_schedule = getattr(args, 'beta_schedule', 'polynomial')
        # AdaSmooth: num_queries is sampling count (K), separate from diagonal L dimension (d)
        # Use config's num_queries (=10) for fair comparison
        # L is always diagonal with d elements, but estimated from K samples
        adasmooth_num_queries = getattr(args, 'adasmooth_num_queries', args.num_queries)
        # AdaSmooth mu: Use same as other ZO methods or slightly smaller
        # With higher beta_init, we can use normal mu
        adasmooth_mu = getattr(args, 'adasmooth_mu', args.mu)
        return AdaSmoothZO(
            params=params,
            lr=1.0,  # Must be 1.0
            num_queries=adasmooth_num_queries,
            mu=adasmooth_mu,
            beta_init=beta_init,
            beta_decay=beta_decay,
            beta_schedule=beta_schedule
        )
    elif name == "adasmooth_multi" or name == "adasmoothzo_multi":
        # AdaSmoothZO for multi-parameter models
        beta_init = getattr(args, 'beta_init', 10.0)  # Higher initial temperature
        beta_decay = getattr(args, 'beta_decay', 0.001)  # Much slower decay
        beta_schedule = getattr(args, 'beta_schedule', 'polynomial')
        adasmooth_num_queries = getattr(args, 'adasmooth_num_queries', args.num_queries)
        adasmooth_mu = getattr(args, 'adasmooth_mu', args.mu)
        return AdaSmoothZO_MultiParam(
            params=params,
            lr=1.0,  # Must be 1.0
            num_queries=adasmooth_num_queries,
            mu=adasmooth_mu,
            beta_init=beta_init,
            beta_decay=beta_decay,
            beta_schedule=beta_schedule
        )
    elif name == "adasmooth_es" or name == "adasmoothes":
        # AdaSmoothES: Complete implementation combining AdaSmooth + SepCMA stability
        # Combines Boltzmann-weighted moment matching with evolution path and CSA
        # Uses diagonal covariance (d elements) estimated from K=num_queries samples
        sigma = getattr(args, 'sigma', args.mu)  # Initial step size
        beta_init = getattr(args, 'beta_init', 10.0)  # Initial temperature
        beta_decay = getattr(args, 'beta_decay', 0.001)  # Temperature decay
        beta_schedule = getattr(args, 'beta_schedule', 'polynomial')
        baseline_type = getattr(args, 'baseline', 'mean')  # Baseline for variance reduction
        ema_alpha = getattr(args, 'ema_alpha', 0.1)  # For EMA baseline

        # Adaptive beta scheduler (new!)
        # Options: 'std', 'std_decay', 'cma_match', 'entropy_target', 'range'
        adaptive_beta = getattr(args, 'adaptive_beta', None)
        adaptive_beta_scheduler = None
        if adaptive_beta is not None:
            # Get scheduler parameters
            c_beta = getattr(args, 'c_beta', 1.0)
            beta_min = getattr(args, 'beta_min', 1e-8)

            if adaptive_beta in ['std', 'std_based']:
                adaptive_beta_scheduler = get_adaptive_beta_scheduler(
                    'std', c_beta=c_beta, beta_min=beta_min
                )
            elif adaptive_beta in ['std_decay', 'adaptive_decay']:
                adaptive_beta_scheduler = get_adaptive_beta_scheduler(
                    'std_decay', c_beta=c_beta, decay_rate=beta_decay, beta_min=beta_min
                )
            elif adaptive_beta in ['cma', 'cma_match']:
                adaptive_beta_scheduler = get_adaptive_beta_scheduler(
                    'cma_match', decay_rate=getattr(args, 'cma_decay', 0.0), beta_min=beta_min
                )
            elif adaptive_beta in ['entropy', 'entropy_target']:
                target_ratio = getattr(args, 'target_eff_ratio', 0.5)
                adaptive_beta_scheduler = get_adaptive_beta_scheduler(
                    'entropy_target', target_eff_ratio=target_ratio, beta_min=beta_min
                )
            elif adaptive_beta in ['range', 'range_based']:
                adaptive_beta_scheduler = get_adaptive_beta_scheduler(
                    'range', c_beta=c_beta, beta_min=beta_min
                )

        return AdaSmoothES(
            params=params,
            sigma=sigma,
            num_queries=args.num_queries,
            beta_init=beta_init,
            beta_decay=beta_decay,
            beta_schedule=beta_schedule,
            baseline=baseline_type,
            ema_alpha=ema_alpha,
            adaptive_beta_scheduler=adaptive_beta_scheduler
        )
    elif name == "adasmooth_es_v2" or name == "adasmoothes_v2":
        # AdaSmoothES v2: Modular version with pluggable divergences and temperature schedules
        # Supports: KL, Reverse KL, χ², Rényi, Tsallis, Huber divergences
        # Supports: constant, linear, exponential, polynomial, cosine, step, adaptive, cyclic schedules
        sigma = getattr(args, 'sigma', args.mu)  # Initial step size
        baseline_type = getattr(args, 'baseline', 'mean')  # Baseline for variance reduction
        ema_alpha = getattr(args, 'ema_alpha', 0.1)  # For EMA baseline

        # Divergence configuration
        divergence_type = getattr(args, 'divergence', 'kl')  # 'kl', 'reverse_kl', 'chi2', 'renyi', 'tsallis', 'huber'
        divergence_kwargs = {}
        if divergence_type == 'renyi':
            divergence_kwargs['alpha'] = getattr(args, 'renyi_alpha', 2.0)
        elif divergence_type == 'tsallis':
            divergence_kwargs['q'] = getattr(args, 'tsallis_q', 2.0)
        elif divergence_type == 'huber':
            divergence_kwargs['delta'] = getattr(args, 'huber_delta', 1.0)

        # Temperature schedule configuration
        temperature_schedule = getattr(args, 'temperature_schedule', 'polynomial')
        temperature_kwargs = {}
        temperature_kwargs['beta_init'] = getattr(args, 'beta_init', 10.0)

        if temperature_schedule in ['exponential', 'polynomial']:
            temperature_kwargs['decay_rate'] = getattr(args, 'beta_decay', 0.001)
            temperature_kwargs['beta_min'] = getattr(args, 'beta_min', 0.01)
        if temperature_schedule == 'polynomial':
            temperature_kwargs['power'] = getattr(args, 'poly_power', 1.0)
        elif temperature_schedule == 'linear':
            temperature_kwargs['beta_min'] = getattr(args, 'beta_min', 0.1)
            temperature_kwargs['total_iterations'] = getattr(args, 'num_iterations', 10000)
        elif temperature_schedule == 'cosine':
            temperature_kwargs['beta_min'] = getattr(args, 'beta_min', 0.1)
            temperature_kwargs['total_iterations'] = getattr(args, 'num_iterations', 10000)
        elif temperature_schedule == 'step':
            temperature_kwargs['step_size'] = getattr(args, 'step_size', 1000)
            temperature_kwargs['gamma'] = getattr(args, 'step_gamma', 0.5)
            temperature_kwargs['beta_min'] = getattr(args, 'beta_min', 0.01)
        elif temperature_schedule == 'cyclic':
            temperature_kwargs['beta_min'] = getattr(args, 'beta_min', 1.0)
            temperature_kwargs['beta_max'] = getattr(args, 'beta_max', 10.0)
            temperature_kwargs['cycle_length'] = getattr(args, 'cycle_length', 500)
            temperature_kwargs['mode'] = getattr(args, 'cyclic_mode', 'triangular')

        return AdaSmoothESv2(
            params=params,
            sigma=sigma,
            num_queries=args.num_queries,
            divergence=divergence_type,
            temperature_schedule=temperature_schedule,
            baseline=baseline_type,
            ema_alpha=ema_alpha,
            divergence_kwargs=divergence_kwargs,
            temperature_kwargs=temperature_kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer name: {name}, available optimizers are: fo, vanilla, rl, zoar, zohs, zohs_expavg, relizo, zoo, reinforce, es, xnes, twopoint, sepcmaes, adasmooth, adasmooth_multi, adasmooth_es, adasmooth_es_v2")