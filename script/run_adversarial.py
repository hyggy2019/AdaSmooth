import copy
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from model.attack import Attack
from optimizer.tools import get_optimizer
from utils.tools import set_seed

def train(model, optimizer, args):
    def closure():
        f = model()
        
        return f

    history = []

    history.append(model().item())
    
    for _ in range(1, args.num_iterations + 1):
        optimizer.zero_grad()
        f = optimizer.step(closure)
        
        if isinstance(f, torch.Tensor):
            f = f.item()
        history.append(f)
    
    return history

def run_adversarial(args):
    seed = args.seed
    model_name = args.model
    dataset_name = args.dataset
    dimension = args.x_dim
    idx = args.idx
    optimizers = args.optimizers
    device = args.device

    set_seed(seed)
    device = torch.device(device)
    print(f"Using device: {device}")

    x_init = torch.randn(dimension, device=device)

    histories = []
    start = time.time()
    for optimizer_name in optimizers:
        model = Attack(copy.deepcopy(x_init), idx=idx)
        
        valid_parameters = [p for n, p in model.named_parameters() if n == "x"]

        optimizer = get_optimizer(optimizer_name, valid_parameters, args)

        start_1 = time.time()
        history = train(model, optimizer, args)
        print(f"{optimizer_name} optimized value: {history[-1]}, Time taken: {time.time() - start_1:.2f} seconds")

        histories.append(history)

    print(f"Total Time taken: {time.time() - start:.2f} seconds")

    # Plotting
    plt.figure(figsize=(10, 6))
    for i, history in enumerate(histories):
        plt.plot(history, label=optimizers[i])
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel(f'{model_name.upper()} Attack Loss Value', fontsize=12)
    plt.title(f'Convergence Comparison -- {model_name} -- {dataset_name}', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.show()

def test_code():
    pass

if __name__ == '__main__':
    test_code()
