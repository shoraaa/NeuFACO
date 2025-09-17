import os
import random

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from math import sqrt

from net import Net
from mfaco import MFACO
from aco import ACO
from utils import load_test_dataset
import time

EPS = 1e-10
START_NODE = None


@torch.no_grad()
def infer_instance(model, pyg_data, distances, n_ants, t_aco_diff, k_sparse, n_runs=1):
    heu_mat = None
    start_time = time.time()
    if model is not None:
        model.eval()
        heu_vec = model(pyg_data)
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS
        heu_mat = heu_mat.cpu()

    return 0, 0, time.time() - start_time

    all_results = []
    all_diversities = []
    all_times = []
    for _ in range(n_runs):
        aco = MFACO(
            distances.cpu(),
            n_ants,
            heuristic=heu_mat.cpu() if heu_mat is not None else heu_mat,
            k_sparse=k_sparse,
        ) if USE_MFACO else ACO(
            distances.cpu(),
            n_ants,
            heuristic=heu_mat.cpu() if heu_mat is not None else heu_mat,
            k_sparse=k_sparse,
        )
        results = torch.zeros(size=(len(t_aco_diff),))
        diversities = torch.zeros(size=(len(t_aco_diff),))
        elapsed_time = 0
        for i, t in enumerate(t_aco_diff):
            results[i], diversities[i], t = aco.run(t, start_node=START_NODE)
            elapsed_time += t
        all_results.append(results)
        all_diversities.append(diversities)
        all_times.append(elapsed_time)
    avg_results = torch.mean(torch.stack(all_results), dim=0)
    avg_diversities = torch.mean(torch.stack(all_diversities), dim=0)
    avg_time = np.mean(all_times)
    return avg_results, avg_diversities, avg_time


@torch.no_grad()
def test(dataset, model, n_ants, t_aco, k_sparse, n_runs=1):
    _t_aco = [0] + t_aco
    t_aco_diff = [_t_aco[i + 1] - _t_aco[i] for i in range(len(_t_aco) - 1)]
    instance_costs = []
    instance_divs = []
    instance_times = []
    for pyg_data, distances in tqdm(dataset, dynamic_ncols=True):
        avg_results, avg_diversities, avg_time = infer_instance(model, pyg_data, distances, n_ants, t_aco_diff, k_sparse, n_runs)
        instance_costs.append(avg_results)
        instance_divs.append(avg_diversities)
        instance_times.append(avg_time)
    instance_costs = torch.stack(instance_costs)
    instance_divs = torch.stack(instance_divs)
    instance_times = np.array(instance_times)
    # Return per-instance averages and overall averages
    return instance_costs, instance_divs, instance_times, torch.mean(instance_costs, dim=0), torch.mean(instance_divs, dim=0), np.mean(instance_times)


def main(ckpt_path, n_nodes, k_sparse, size=None, n_ants=None, n_iter=1000, guided_exploration=False, seed=0, test_name="", n_runs=1, starting_instance_idx=0):
    test_list = load_test_dataset(n_nodes, k_sparse, DEVICE, start_node=START_NODE)
    
    # Calculate end index based on starting index and size
    if size is not None:
        end_idx = starting_instance_idx + size
        test_list = test_list[starting_instance_idx:end_idx]
        original_indices = list(range(starting_instance_idx, end_idx))
    else:
        test_list = test_list[starting_instance_idx:]
        original_indices = list(range(starting_instance_idx, len(test_list) + starting_instance_idx))

    if n_ants is None:
        n_ants = int(4 * sqrt(n_nodes))
        n_ants = max(64, ((n_ants + 63) // 64) * 64)

    t_aco = list(range(1, n_iter + 1))
    print("problem scale:", n_nodes)
    print("checkpoint:", ckpt_path)
    print("number of instances:", size)
    print("device:", 'cpu' if DEVICE == 'cpu' else DEVICE+"+cpu" )
    print("n_ants:", n_ants)
    print("seed:", seed)
    print("n_runs:", n_runs)

    if ckpt_path is not None:
        net = Net(gfn=True, Z_out_dim=2 if guided_exploration else 1, start_node=START_NODE).to(DEVICE) if GFACS else \
              Net(gfn=False, Z_out_dim=1, start_node=START_NODE).to(DEVICE) if PPO else \
              Net(gfn=False, value_head=True, start_node=START_NODE).to(DEVICE) 
        net.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    else:
        net = None
    instance_costs, instance_divs, instance_times, avg_cost, avg_diversity, duration = test(test_list, net, n_ants, t_aco, k_sparse, n_runs)
    print('average inference time: ', duration)
    for i, t in enumerate(t_aco):
        print(f"T={t}, avg. cost {avg_cost[i]}, avg. diversity {avg_diversity[i]}")

    # Save result in directory that contains model_file
    filename = os.path.splitext(os.path.basename(ckpt_path))[0] if ckpt_path is not None else 'none'
    dirname = f'../tsp/results_test/{filename}'
    os.makedirs(dirname, exist_ok=True)

    result_filename = f"test_result_ckpt{filename}-tsp{n_nodes}-ninst{size}-{ACOALG}-nants{n_ants}-niter{n_iter}-nruns{n_runs}-seed{seed}{'-'+test_name if test_name else ''}"
    
        # Save iteration-wise results to separate CSV
    iteration_results = pd.DataFrame(columns=['T', 'avg_cost', 'avg_diversity'])
    for i, t in enumerate(t_aco):
        iteration_results = pd.concat([iteration_results, pd.DataFrame({
            'T': [t],
            'avg_cost': [avg_cost[i].item()],
            'avg_diversity': [avg_diversity[i].item()]
        })], ignore_index=True)
    iteration_results.to_csv(os.path.join(dirname, result_filename + "_iterations.csv"), index=False)
    
    # Save per-instance averages (keep original format)
    results = pd.DataFrame(columns=['instance', 'mean_cost', 'min_cost', 'max_cost', 'avg_time'])
    for idx in range(len(test_list)):
        original_idx = original_indices[idx]
        mean_cost_val = instance_costs[idx].mean().item()
        min_cost_val = instance_costs[idx].min().item()
        max_cost_val = instance_costs[idx].max().item()
        avg_time_val = instance_times[idx]
        results = pd.concat([results, pd.DataFrame({
            'instance': [original_idx],
            'mean_cost': [mean_cost_val],
            'min_cost': [min_cost_val],
            'max_cost': [max_cost_val],
            'avg_time': [avg_time_val]
        })], ignore_index=True)
    results.to_csv(os.path.join(dirname, result_filename + ".csv"), index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", type=int, help="Problem scale")
    parser.add_argument("-k", "--k_sparse", type=int, default=None, help="k_sparse")
    parser.add_argument("-p", "--path", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("-i", "--n_iter", type=int, default=100, help="Number of iterations of ACO to run")
    parser.add_argument("-n", "--n_ants", type=int, default=100, help="Number of ants")
    parser.add_argument("-d", "--device", type=str,
                        default=("cuda:0" if torch.cuda.is_available() else "cpu"), 
                        help="The device to train NNs")
    parser.add_argument("-s", "--size", type=int, default=None, help="Number of instances to test")
    parser.add_argument("-si", "--starting_instance_idx", type=int, default=0, help="Starting instance index for inference")
    parser.add_argument("--test_name", type=str, default="", help="Name of the test")
    parser.add_argument("-r", "--n_runs", type=int, default=1, help="Number of runs per instance")
    ### GFACS
    parser.add_argument("--disable_guided_exp", action='store_true', help='True for model w/o guided exploration.')
    ### Seed
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    ### ACO
    parser.add_argument("--aco", type=str, default="AS", choices=["AS", "ELITIST", "MAXMIN", "RANK"], help="ACO algorithm")
    parser.add_argument("-g", "--gfacs", action='store_true', help="Loading GFACS model")
    parser.add_argument("--ppo", action='store_true', help="Loading PPO model")
    parser.add_argument("--mfaco", action='store_true', help="Use mfaco model")
    args = parser.parse_args()

    if args.k_sparse is None:
        args.k_sparse = args.nodes // 10

    DEVICE = args.device if torch.cuda.is_available() else 'cpu'
    ACOALG = args.aco
    GFACS = args.gfacs
    PPO = args.ppo
    USE_MFACO = args.mfaco

    # seed everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.path is not None and not os.path.isfile(args.path):
        print(f"Checkpoint file '{args.path}' not found!")
        exit(1)

    main(
        args.path,
        args.nodes,
        args.k_sparse,
        args.size,
        args.n_ants,
        args.n_iter,
        not args.disable_guided_exp,
        args.seed,
        args.test_name,
        args.n_runs,
        args.starting_instance_idx
    )