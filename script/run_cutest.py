import copy
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from model.synthetic_functions import get_synthetic_funcs
from optimizer.relizo import LIZO, _backtracking
from utils.tools import set_seed

NUM_Q = 10

def true_gradient_optimization(obj, step_size, num_iterations, cutest=False):
    history = []
    m = torch.zeros_like(x_init)  # First moment vector
    v = torch.zeros_like(x_init)  # Second moment vector
    # beta1 = 0.9
    beta1 = 0.0
    beta2 = 0.99
    epsilon = 1e-8
    for t in range(1, num_iterations + 1):
        x = obj.x
        f = obj()
        history.append(f.item())
        if not cutest:
            if x.grad is not None: 
                x.grad.zero_()
            f.backward()  # Compute the gradient

        with torch.no_grad():
            grad = x.grad
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t)
            # v_hat = v / (1 - beta2 ** t)
            v_hat = torch.tensor(1.0, device=x.device)
            x.add_(- step_size * m_hat / (torch.sqrt(v_hat) + epsilon))
    
    return history, obj()

@torch.no_grad()
def finite_difference_optimization(obj, step_size, num_iterations, mu=0.01):
    history = []
    m = torch.zeros_like(x_init)  # First moment vector
    v = torch.zeros_like(x_init)  # Second moment vector
    # beta1 = 0.9
    beta1 = 0.0
    beta2 = 0.99
    epsilon = 1e-8
    for t in range(1, num_iterations + 1):
        x = obj.x
        f = obj()
        history.append(f.item())
        grad_est = 0
        ds = []
        fxs = []
        for _ in range(NUM_Q):
            random_direction = torch.randn(len(x), device=x.device)
            # random_direction /= np.linalg.norm(random_direction)
            ds.append(random_direction)
            x.add_(mu * random_direction)
            f_x_plus_h = obj()
            fxs.append(f_x_plus_h.item())
            x.add_(-mu * random_direction)
        
        fxs = torch.tensor(fxs, device=x.device)
        # fxs_mean = np.mean(fxs) # a large mu should use this one, while a smaller one should use the following one
        fxs_mean = obj()
        for d, f_x_plus_h in zip(ds, fxs):
            grad_est += (f_x_plus_h - fxs_mean) / mu * d
        grad_est /= NUM_Q
        m = beta1 * m + (1 - beta1) * grad_est
        v = beta2 * v + (1 - beta2) * (grad_est ** 2)
        m_hat = m / (1 - beta1 ** t)
        # v_hat = v / (1 - beta2 ** t)
        v_hat = torch.tensor(1.0, device=x.device)
        x.add_(- step_size * m_hat / (torch.sqrt(v_hat) + epsilon))
    
    return history, obj()

@torch.no_grad()
def reinforce_optimization(obj, step_size, num_iterations, variance):
    """
    Standard REINFORCE algorithm for optimization.

    Args:
        func: The function to minimize.
        x_init: Initial guess for the minimizer.
        step_size: Step size for the gradient update.
        num_iterations: Number of iterations to run.
        variance: Variance of the Gaussian noise added to the parameter vector.

    Returns:
        A tuple containing the history of function values and the final parameter vector.
    """
    history = []
    m = torch.zeros_like(x_init)  # First moment vector
    v = torch.zeros_like(x_init)  # Second moment vector
    # beta1 = 0.9
    beta1 = 0.0
    beta2 = 0.99
    epsilon = 1e-8
    std = torch.sqrt(torch.tensor(variance))
    
    for t in range(1, num_iterations + 1):
        x_mean = obj.x
        f = obj()
        history.append(f.item())  # Store function value at x_mean
        grad_est = 0
        rewards = []
        log_probs = []
        log_prob_square = torch.zeros_like(x_mean)
        
        for _ in range(NUM_Q + 1):
            # Sample action (noise) from a Gaussian distribution
            noise = std * torch.randn(len(x_mean), device=x_mean.device)
            x_mean.add_(noise)
            reward = obj()
            x_mean.add_(-noise)

            rewards.append(reward)
            grad_log_prob = noise / variance
            # grad_log_prob = grad_log_prob / dist.pdf(noise) # trpo does not work well
            # log_prob_square += grad_log_prob ** 2
            log_probs.append(grad_log_prob)
            
        rewards = torch.tensor(rewards, device=x_mean.device)
        log_probs = torch.stack(log_probs)
        
        inds = torch.argsort(rewards)
        rewards[inds] = 2 * torch.arange(NUM_Q + 1, device=rewards.device, dtype=rewards.dtype)
        # rewards = np.exp(rewards)
        rewards = (rewards - rewards.mean())
        # rewards = np.sign(rewards)
        # rewards = rewards - func(x_mean)
        
        for reward, log_prob in zip(rewards, log_probs):
            grad_est += reward * log_prob # Directly use log_prob, not noise itself
            # grad_est += reward * log_prob / log_prob_square
            
        grad_est /= NUM_Q
        m = beta1 * m + (1 - beta1) * grad_est
        v = beta2 * v + (1 - beta2) * (grad_est ** 2)
        m_hat = m / (1 - beta1 ** t)
        # v_hat = v / (1 - beta2 ** t)
        v_hat = torch.tensor(1.0, device=x_mean.device)
        x_mean.add_(- step_size * m_hat / (torch.sqrt(v_hat) + epsilon))

    return history, obj()

@torch.no_grad()
def reinforce_optimization_w_history(obj, step_size, num_iterations, variance):
    """
    Standard REINFORCE algorithm for optimization with history.

    Args:
        func: The function to minimize.
        x_init: Initial guess for the minimizer.
        step_size: Step size for the gradient update.
        num_iterations: Number of iterations to run.
        variance: Variance of the Gaussian noise added to the parameter vector.

    Returns:
        A tuple containing the history of function values and the final parameter vector.
    """
    history = []
    m = torch.zeros_like(x_init)  # First moment vector
    v = torch.zeros_like(x_init)  # Second moment vector
    # beta1 = 0.9
    beta1 = 0.0
    beta2 = 0.99
    epsilon = 1e-8
    std = torch.sqrt(torch.tensor(variance))
    prev = []
    
    for t in range(1, num_iterations + 1):
        x_mean = obj.x
        f = obj()
        history.append(f.item())  # Store function value at x_mean
        grad_est = 0
        rewards = []
        log_probs = []
        post = []
        
        for _ in range(NUM_Q + 1):
            # Sample action (noise) from a Gaussian distribution
            noise = std * torch.randn(len(x_mean), device=x_mean.device)
            x_mean.add_(noise)
            reward = obj()
            x_mean.add_(-noise)

            rewards.append(reward)
            grad_log_prob = noise / variance
            log_probs.append(grad_log_prob)
            
            # Calculate log probability of noise under Gaussian distribution
            log_prob_sum = torch.sum(torch.distributions.Normal(0, std).log_prob(noise))
            post.append([noise, reward, log_prob_sum])
        
        if t == 1:
            rewards = torch.tensor(rewards, device=x_mean.device)
            log_probs = torch.stack(log_probs)
            ratios = torch.ones_like(rewards)
        else:
            # rewards = np.array(rewards + [p[1] for p in prev])
            rewards = torch.tensor([r for r in rewards] + [p[1] for p in prev], device=x_mean.device)
            prev_log_probs = [(p[0])/variance for p in prev]
            # log_probs = np.array(log_probs + [(p[0] - x_mean)/variance for p in prev])
            log_probs = torch.stack(log_probs + prev_log_probs)
            # ratios = np.array([1.0 for _ in range(NUM_Q + 1)] + [np.exp(np.sum(np.log(stats.norm.pdf(p[0], 0, np.sqrt(variance)))) - p[2]) for p in prev])
            # ratios = np.clip(ratios, 0.0, 1.3) # this is exactly PPO

        prev += post
        prev = prev[-15*(NUM_Q+1):]
        # prev = []
        
        # print("before", rewards)
        # inds = np.argsort(rewards)
        # rewards[inds] = np.arange(len(rewards))
        # print("after", rewards)
        rewards = (rewards - rewards.mean()) #/ np.std(rewards)
        # rewards = rewards - func(x_mean)
        for k in range(len(rewards)):
            grad_est += rewards[k] * log_probs[k] #* ratios[k]  # Directly use log_prob, not noise itself
        
        grad_est /= len(rewards)
        m = beta1 * m + (1 - beta1) * grad_est
        v = beta2 * v + (1 - beta2) * (grad_est ** 2)
        m_hat = m / (1 - beta1 ** t)
        # v_hat = v / (1 - beta2 ** t)
        v_hat = torch.tensor(1.0, device=x_mean.device)
        x_mean.add_(- step_size * m_hat / (torch.sqrt(v_hat) + epsilon))

    return history, obj()

@torch.no_grad()
def relizo_optimization(obj, step_size, num_iterations, cutest=False):
    history = []
    optimizer = LIZO(
        obj.parameters(),
        lr=step_size,
        weight_decay=0.0,
        num_sample_per_step=NUM_Q,
        reuse_distance_bound=2*step_size,
        max_reuse_rate=0.5,
        orthogonal_sample=False,
        # line_search_fn=partial(_backtracking),
        strict_lr=True,
    )

    def _closure():
        if cutest:
            return obj(no_grad=cutest)
        else:
            return obj()

    for t in range(1, num_iterations + 1):
        f = obj()
        history.append(f.item())  # Store function value at x_mean
        optimizer.zero_grad()
        optimizer.step(_closure)
    
    return history, obj()



if __name__ == '__main__':
    dimension = 1000
    set_seed(456)
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    x_init = torch.randn(dimension, device=device) * 5
    step_size = 0.0001
    num_iterations = 2000
    variance = 0.1
    # name = "levy"
    name = "cutest_BOXPOWER"
    obj = get_synthetic_funcs(name, x_init)

    start = time.time()
    true_history, true_f_opt = true_gradient_optimization(copy.deepcopy(obj), step_size, num_iterations, cutest=name.startswith("cutest_"))
    print(f"True Gradient Optimized Value: {true_f_opt.item()}, Time taken: {time.time() - start:.2f} seconds")

    start_1 = time.time()
    fd_history, fd_f_opt = finite_difference_optimization(copy.deepcopy(obj), step_size, num_iterations, mu=torch.sqrt(torch.tensor(variance)))
    print(f"Finite Difference Optimized Value: {fd_f_opt.item()}, Time taken: {time.time() - start_1:.2f} seconds")
    
    start_1 = time.time()
    reinforce_history, reinforce_f_opt = reinforce_optimization(copy.deepcopy(obj), step_size, num_iterations, variance)
    print(f"REINFORCE Optimized Value: {reinforce_f_opt.item()}, Time taken: {time.time() - start_1:.2f} seconds")
    
    start_1 = time.time()
    reinforce_history_w_history, reinforce_f_opt_w_history = reinforce_optimization_w_history(copy.deepcopy(obj), step_size, num_iterations, variance)
    print(f"REINFORCE w/ History Optimized Value: {reinforce_f_opt_w_history.item()}, Time taken: {time.time() - start_1:.2f} seconds")

    start_1 = time.time()
    relizo_history, relizo_f_opt = relizo_optimization(copy.deepcopy(obj), step_size, num_iterations, cutest=name.startswith("cutest_"))
    print(f"Relizo Optimized Value: {relizo_f_opt.item()}, Time taken: {time.time() - start_1:.2f} seconds")
    
    print(f"Time comsumption: {time.time() - start:.2f} seconds")

    plt.figure(figsize=(10, 6))
    plt.plot(true_history, label='True Gradient')
    plt.plot(fd_history, label='Finite Difference')
    plt.plot(reinforce_history, label='REINFORCE')
    plt.plot(reinforce_history_w_history, label='REINFORCE w/ History')
    plt.plot(relizo_history, label='Relizo') 
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel(f'{name.capitalize()} Function Value', fontsize=12)
    plt.title(f'Convergence Comparison -- {name} -- d = {dimension}', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.show()
