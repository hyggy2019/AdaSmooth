import time
import numpy as np
import torch
from model.attack import Attack
from utils import set_seed, get_optimizer
import os

def train(model, optimizer, args):
    history = []

    history.append(model().item())
    
    for _ in range(1, args.num_iterations + 1):
        optimizer.zero_grad()
        f = optimizer.step(model)
        
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
    
    device = torch.device(device)
    print(f"Using device: {device}")

    histories = []
    start = time.time()
    for optimizer_name in optimizers:
        set_seed(seed)
        x_init = torch.randn(dimension, device=device)
        model = Attack(x_init, idx=idx)
        
        valid_parameters = [p for n, p in model.named_parameters() if n == "x"]

        optimizer = get_optimizer(optimizer_name, valid_parameters, args)

        start_1 = time.time()
        history = train(model, optimizer, args)
        print(f"{optimizer_name} optimized value: {history[-1]}, Time taken: {time.time() - start_1:.2f} seconds")

        histories.append(history)

        # Save the history to a file
        tag = f"{args.dataset}_{optimizer_name}_{args.update_rule}_ni{args.num_iterations}_lr{args.lr}_nq{args.num_queries}_mu{args.mu}_nh{args.num_histories}_s{seed}"

        # create a directory if it doesn't exist
        os.makedirs("results/attack", exist_ok=True)
        torch.save(history, f"results/attack/{tag}.pt")

    print(f"Total Time taken: {time.time() - start:.2f} seconds")

def test_code():
    pass

if __name__ == '__main__':
    test_code()
