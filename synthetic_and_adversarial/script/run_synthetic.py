import copy
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from model.synthetic_functions import get_synthetic_funcs
from utils import set_seed, get_optimizer
import os

def train(func, optimizer, args):
    history = []
    with torch.no_grad():
        history.append(func().item())
    
    for _ in range(args.num_iterations):
        optimizer.zero_grad()
        f = optimizer.step(func)
        
        if isinstance(f, torch.Tensor):
            f = f.item()
        history.append(f)
    
    return history

def run_synthetic(args):
    seed = args.seed
    func_name = args.func_name
    dimension = args.dimension
    optimizers = args.optimizers
    
    device = torch.device("cpu")
    print(f"Using device: {device}")

    histories = []
    start = time.time()
    for optimizer_name in optimizers:
        set_seed(seed)
        x_init = torch.randn(dimension, device=device)

        func = get_synthetic_funcs(func_name, x_init)
        optimizer = get_optimizer(optimizer_name, func.parameters(), args)

        start_1 = time.time()
        history = train(func, optimizer, args)

        print(f"{optimizer_name} optimized value: {history[-1]}, Time taken: {time.time() - start_1:.2f} seconds")

        histories.append(history)

        # Save the history to a file
        tag = f"{func_name}_{optimizer_name}_{args.update_rule}_d{dimension}_ni{args.num_iterations}_lr{args.lr}_nq{args.num_queries}_mu{args.mu}_nh{args.num_histories}_s{seed}" 
        # tag = f"{func_name}_{optimizer_name}_{args.update_rule}_{args.baseline}_d{dimension}_ni{args.num_iterations}_lr{args.lr}_nq{args.num_queries}_mu{args.mu}_s{seed}" # for baseline experiments

        # create a directory if it doesn't exist
        os.makedirs("results/synthetic", exist_ok=True)
        torch.save(history, f"results/synthetic/{tag}.pt")

    print(f"Total Time taken: {time.time() - start:.2f} seconds")

def test_code():
    pass

if __name__ == '__main__':
    test_code()
