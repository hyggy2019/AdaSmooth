import copy
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from model.synthetic_functions import get_synthetic_funcs
from optimizer.tools import get_optimizer
from utils.tools import set_seed

def train(func, optimizer, args):
    def closure():
        f = func()
        
        return f

    history = []
    with torch.no_grad():
        history.append(func().item())
    
    for _ in range(1, args.num_iterations + 1):
        optimizer.zero_grad()
        f = optimizer.step(closure)
        history.append(f.item())
    
    return history

def run_synthetic(args):
    seed = args.seed
    func_name = args.func_name
    dimension = args.dimension
    optimizers = args.optimizers

    set_seed(seed)
    device = torch.device("cpu")
    print(f"Using device: {device}")

    x_init = torch.randn(dimension, device=device) * 5

    histories = []
    start = time.time()
    for optimizer_name in optimizers:
        func = get_synthetic_funcs(func_name, x_init)
        optimizer = get_optimizer(optimizer_name, func.parameters(), args)

        start_1 = time.time()
        history = train(func, optimizer, args)
        print(f"{optimizer_name} optimized value: {history[-1]}, Time taken: {time.time() - start_1:.2f} seconds")

        histories.append(history)

    print(f"Total Time taken: {time.time() - start:.2f} seconds")

    # Plotting
    plt.figure(figsize=(10, 6))
    for i, history in enumerate(histories):
        plt.plot(history, label=optimizers[i])
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel(f'{func_name.capitalize()} Function Value', fontsize=12)
    plt.title(f'Convergence Comparison -- {func_name} -- d = {dimension}', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.show()

def test_code():
    pass

if __name__ == '__main__':
    test_code()
