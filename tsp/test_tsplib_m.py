import os
import random
import json

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

from net import Net
from mfaco import MFACO
from aco import ACO_NP, ACO
from utils import load_tsplib_dataset


EPS = 1e-10
START_NODE = None


@torch.no_grad()
def infer_instance(model, pyg_data, distances, n_ants, t_aco_diff, n_runs=1):
    if model is not None:
        model.eval()
        heu_vec = model(pyg_data)
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS
    else:
        heu_mat = None

    all_run_results = []
    all_run_paths = []
    total_elapsed_time = 0
    
    for run_idx in range(n_runs):
        aco = MFACO(
            distances.cpu(),
            n_ants,
            heuristic=heu_mat.cpu() if heu_mat is not None else heu_mat,
        ) if USE_MFACO else ACO(
            distances.cpu(),
            n_ants,
            heuristic=heu_mat.cpu() if heu_mat is not None else heu_mat,
            k_sparse=len(distances)/10,
        )

        results = torch.zeros(size=(len(t_aco_diff),))
        elapsed_time = 0
        for i, t in enumerate(t_aco_diff):
            cost, _, t = aco.run(t, start_node=START_NODE)
            results[i] = cost
            elapsed_time += t
            if n_runs == 1:  # Only print detailed info for single runs
                print(f"Iteration {i+1}/{len(t_aco_diff)}: Cost = {cost}, Time = {t:.2f}s")
        
        all_run_results.append(results)
        all_run_paths.append(aco.shortest_path)
        total_elapsed_time += elapsed_time
        
        if n_runs > 1:
            print(f"Run {run_idx+1}/{n_runs}: Best cost = {results[-1]:.2f}, Time = {elapsed_time:.2f}s")
    
    # Calculate average results across runs
    avg_results = torch.mean(torch.stack(all_run_results), dim=0)
    # Use the best path from the run that achieved the best final result
    best_run_idx = torch.argmin(torch.stack([results[-1] for results in all_run_results]))
    best_path = all_run_paths[best_run_idx]
    avg_elapsed_time = total_elapsed_time / n_runs
    
    return avg_results, best_path, avg_elapsed_time, all_run_results


def infer_instance_np(model, pyg_data, distances, n_ants, t_aco_diff, n_runs=1):
    """
    NumPy version of infer_instance using ACO_NP instead of MFACO.
    
    Args:
        model: Neural network model (can be None for vanilla ACO)
        pyg_data: PyTorch Geometric data (for model inference)
        distances: Distance matrix as numpy array or torch tensor
        n_ants: Number of ants
        t_aco_diff: List of iteration counts for each ACO run
        n_runs: Number of independent runs
    
    Returns:
        avg_results: Average results across runs as numpy array
        best_path: Best path found as numpy array
        avg_elapsed_time: Average elapsed time
        all_run_results: All results from all runs
    """
    # Convert distances to numpy if it's a torch tensor
    if isinstance(distances, torch.Tensor):
        distances_np = distances.cpu().numpy()
    else:
        distances_np = distances
    
    # Handle heuristic matrix
    heu_mat_np = None
    if model is not None:
        with torch.no_grad():
            model.eval()
            heu_vec = model(pyg_data)
            heu_mat = model.reshape(pyg_data, heu_vec) + EPS
            heu_mat_np = heu_mat.cpu().numpy()

    all_run_results = []
    all_run_paths = []
    total_elapsed_time = 0
    
    for run_idx in range(n_runs):
        aco = ACO_NP(
            distances_np,
            n_ants,
            heuristic=heu_mat_np,
        )

        results = np.zeros(len(t_aco_diff))
        elapsed_time = 0
        for i, t in enumerate(t_aco_diff):
            cost, _, t = aco.run(t, start_node=START_NODE)
            results[i] = cost
            elapsed_time += t
            if n_runs == 1:  # Only print detailed info for single runs
                print(f"Iteration {i+1}/{len(t_aco_diff)}: Cost = {cost}, Time = {t:.2f}s")
        
        all_run_results.append(results)
        all_run_paths.append(aco.shortest_path)
        total_elapsed_time += elapsed_time
        
        if n_runs > 1:
            print(f"Run {run_idx+1}/{n_runs}: Best cost = {results[-1]:.2f}, Time = {elapsed_time:.2f}s")
    
    # Calculate average results across runs
    avg_results = np.mean(np.stack(all_run_results), axis=0)
    # Use the best path from the run that achieved the best final result
    best_run_idx = np.argmin([results[-1] for results in all_run_results])
    best_path = all_run_paths[best_run_idx]
    avg_elapsed_time = total_elapsed_time / n_runs
    
    return avg_results, best_path, avg_elapsed_time, all_run_results


@torch.no_grad()
def test(dataset, scale_list, model, n_ants, t_aco, n_runs=1):
    _t_aco = [0] + t_aco
    t_aco_diff = [_t_aco[i + 1] - _t_aco[i] for i in range(len(_t_aco) - 1)]

    results_list = []
    best_paths = []
    all_runs_data = []  # Store all run data for statistical analysis
    instance_times = []  # Store average time per instance
    sum_times = 0
    instance_idx = 0
    for (pyg_data, distances), scale in tqdm(zip(dataset, scale_list)):
        ceiled_distances = (distances * scale).ceil()
        results, best_path, elapsed_time, all_run_results = infer_instance(model, pyg_data, ceiled_distances, n_ants, t_aco_diff, n_runs)
        results_list.append(results)
        best_paths.append(best_path)
        all_runs_data.append(all_run_results)
        instance_times.append(elapsed_time)
        sum_times += elapsed_time
        instance_idx += 1
        print(f"Instance {instance_idx}: Best cost = {results[-1]:.2f}, Avg Time = {elapsed_time:.2f}s")
    return results_list, best_paths, sum_times / len(dataset), all_runs_data, instance_times


def test_np(dataset, scale_list, model, n_ants, t_aco, n_runs=1):
    """
    NumPy version of test function using ACO_NP instead of MFACO.
    
    Args:
        dataset: Dataset containing (pyg_data, distances) tuples
        scale_list: List of scaling factors for each instance
        model: Neural network model (can be None for vanilla ACO)
        n_ants: Number of ants
        t_aco: List of cumulative iteration counts
        n_runs: Number of independent runs
    
    Returns:
        results_list: List of results for each instance
        best_paths: List of best paths for each instance
        avg_time: Average time per instance
        all_runs_data: All run data for statistical analysis
        instance_times: Time for each instance
    """
    _t_aco = [0] + t_aco
    t_aco_diff = [_t_aco[i + 1] - _t_aco[i] for i in range(len(_t_aco) - 1)]

    results_list = []
    best_paths = []
    all_runs_data = []  # Store all run data for statistical analysis
    instance_times = []  # Store average time per instance
    sum_times = 0
    for (pyg_data, distances), scale in tqdm(zip(dataset, scale_list)):
        # Convert distances to numpy and apply scaling
        if isinstance(distances, torch.Tensor):
            distances_np = distances.cpu().numpy()
        else:
            distances_np = distances
        ceiled_distances = np.ceil(distances_np * scale)
        
        results, best_path, elapsed_time, all_run_results = infer_instance_np(model, pyg_data, ceiled_distances, n_ants, t_aco_diff, n_runs)
        results_list.append(results)
        best_paths.append(best_path)
        all_runs_data.append(all_run_results)
        instance_times.append(elapsed_time)
        sum_times += elapsed_time
    return results_list, best_paths, sum_times / len(dataset), all_runs_data, instance_times


def make_tsplib_data(filename, episode):
    instance_data = []
    cost = []
    instance_name = []
    for line in open(filename, "r").readlines()[episode: episode + 1]:
        line = line.rstrip("\n")
        line = line.replace('[', '')
        line = line.replace(']', '')
        line = line.replace('\'', '')
        line = line.split(sep=',')

        line_data = np.array(line[2:], dtype=float).reshape(-1, 2)
        instance_data.append(line_data)
        cost.append(np.array(line[1], dtype=float))
        instance_name.append(np.array(line[0], dtype=str))
    instance_data = np.array(instance_data)  
    cost = np.array(cost)
    instance_name = np.array(instance_name)
    
    return instance_data, cost, instance_name


def main(ckpt_path, n_nodes, k_sparse_factor=10, n_ants=None, n_iter=10, guided_exploration=False, seed=0, n_runs=1, use_numpy=False):
    test_list, scale_list, name_list = load_tsplib_dataset(n_nodes, k_sparse_factor, DEVICE, start_node=START_NODE)

    if n_ants is None:
        n_ants = int(4 * np.sqrt(n_nodes))
        n_ants = max(64, ((n_ants + 63) // 64) * 64)

    t_aco = list(range(1, n_iter + 1))
    print("problem scale:", n_nodes)
    print("checkpoint:", ckpt_path)
    print("number of instances:", len(test_list))
    print("device:", 'cpu' if DEVICE == 'cpu' else DEVICE+"+cpu" )
    print("n_ants:", n_ants)
    print("n_runs:", n_runs)
    print("seed:", seed)
    print("using numpy:", use_numpy)

    if ckpt_path is not None:
        net = Net(gfn=True, Z_out_dim=2 if guided_exploration else 1, start_node=START_NODE).to(DEVICE) if GFACS else \
              Net(gfn=False, Z_out_dim=1, start_node=START_NODE).to(DEVICE) if PPO else \
              Net(gfn=False, value_head=True, start_node=START_NODE).to(DEVICE) 
            
        net.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    else:
        net = None
    
    # Choose between NumPy and PyTorch implementation
    if use_numpy:
        results_list, best_paths, duration, all_runs_data, instance_times = test_np(test_list, scale_list, net, n_ants, t_aco, n_runs)
    else:
        results_list, best_paths, duration, all_runs_data, instance_times = test(test_list, scale_list, net, n_ants, t_aco, n_runs)

    ### results_list is not consistent with the lengths calculated by below code. IDK why...
    ### Reload the original TSPlib data for cost calculation, as they rounds up the distances
    tsplib_data_list = [make_tsplib_data("../data/tsp/tsplib/TSPlib_70instances.txt", i) for i in range(70)]
    tsplib_instances = [(dat[0][0], dat[1], dat[2][0]) for dat in tsplib_data_list if dat[2][0] in name_list]
    assert len(tsplib_instances) == len(best_paths)

    # Load best-known values from JSON file
    try:
        with open("best-known.json", "r") as f:
            best_known = json.load(f)
    except FileNotFoundError:
        print("Warning: best-known.json not found. Error calculations will be skipped.")
        best_known = {}

    # Create DataFrame with columns for statistics if multiple runs
    if n_runs > 1:
        columns = ["Length_Mean", "Length_Std", "Length_Min", "Length_Max", "Length_Best", "Avg_Time", 
                  "Best_Known", "Error_Mean_%", "Error_Min_%", "Error_Max_%", "Error_Best_%"]
    else:
        columns = ["Length", "Avg_Time", "Best_Known", "Error_%"]
    
    results_df = pd.DataFrame(columns=columns)
    
    for i, (inst, path) in enumerate(zip(tsplib_instances, best_paths)):
        coords, _, tsp_name = inst
        
        if n_runs > 1:
            # Calculate statistics across all runs for this instance
            all_lengths = []
            for run_results in all_runs_data[i]:
                # Use the final cost from each run
                all_lengths.append(float(run_results[-1]))
            
            # Calculate tour length for the best path (already selected as the best across runs)
            # Handle both numpy arrays and torch tensors
            if isinstance(path, torch.Tensor):
                path_np = path.cpu().numpy()
            else:
                path_np = path
            tour_coords = np.concatenate([coords[path_np], coords[path_np[0]].reshape(1, 2)], axis=0)
            tour_length = np.linalg.norm(tour_coords[1:] - tour_coords[:-1], axis=1)
            tour_length = np.ceil(tour_length).astype(int)
            best_length = np.sum(tour_length)
            
            # Store statistics
            results_df.loc[tsp_name, "Length_Mean"] = np.mean(all_lengths)
            results_df.loc[tsp_name, "Length_Std"] = np.std(all_lengths)
            results_df.loc[tsp_name, "Length_Min"] = np.min(all_lengths)
            results_df.loc[tsp_name, "Length_Max"] = np.max(all_lengths)
            results_df.loc[tsp_name, "Length_Best"] = best_length
            results_df.loc[tsp_name, "Avg_Time"] = instance_times[i]
            
            # Calculate error percentages based on best-known value for all length metrics
            if tsp_name in best_known:
                best_known_value = best_known[tsp_name]
                results_df.loc[tsp_name, "Best_Known"] = best_known_value
                error_mean = ((np.mean(all_lengths) - best_known_value) / best_known_value) * 100
                error_min = ((np.min(all_lengths) - best_known_value) / best_known_value) * 100
                error_max = ((np.max(all_lengths) - best_known_value) / best_known_value) * 100
                error_best = ((best_length - best_known_value) / best_known_value) * 100
                
                results_df.loc[tsp_name, "Error_Mean_%"] = error_mean
                results_df.loc[tsp_name, "Error_Min_%"] = error_min
                results_df.loc[tsp_name, "Error_Max_%"] = error_max
                results_df.loc[tsp_name, "Error_Best_%"] = error_best
            else:
                results_df.loc[tsp_name, "Best_Known"] = "N/A"
                results_df.loc[tsp_name, "Error_Mean_%"] = "N/A"
                results_df.loc[tsp_name, "Error_Min_%"] = "N/A"
                results_df.loc[tsp_name, "Error_Max_%"] = "N/A"
                results_df.loc[tsp_name, "Error_Best_%"] = "N/A"
        else:
            # Original single run behavior
            # Handle both numpy arrays and torch tensors
            if isinstance(path, torch.Tensor):
                path_np = path.cpu().numpy()
            else:
                path_np = path
            tour_coords = np.concatenate([coords[path_np], coords[path_np[0]].reshape(1, 2)], axis=0)
            tour_length = np.linalg.norm(tour_coords[1:] - tour_coords[:-1], axis=1)
            tour_length = np.ceil(tour_length).astype(int)
            tsp_results = np.sum(tour_length)
            results_df.loc[tsp_name, "Length"] = tsp_results
            results_df.loc[tsp_name, "Avg_Time"] = instance_times[i]
            
            # Calculate error percentage based on best-known value
            if tsp_name in best_known:
                best_known_value = best_known[tsp_name]
                results_df.loc[tsp_name, "Best_Known"] = best_known_value
                error_pct = ((tsp_results - best_known_value) / best_known_value) * 100
                results_df.loc[tsp_name, "Error_%"] = error_pct
            else:
                results_df.loc[tsp_name, "Best_Known"] = "N/A"
                results_df.loc[tsp_name, "Error_%"] = "N/A"

    print('average inference time: ', duration)
    # Save result in directory that contains model_file
    filename = os.path.splitext(os.path.basename(ckpt_path))[0] if ckpt_path is not None else 'none'
    dirname = f'../tsp/results_testlib/{filename}'
    os.makedirs(dirname, exist_ok=True)

    result_filename = f"test_result_ckpt{filename}-tsplib{n_nodes}-nants{n_ants}-niter{n_iter}-nruns{n_runs}-seed{seed}"
    results_df.to_csv(os.path.join(dirname, f"{result_filename}.csv"), index=True)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", type=int, help="Problem scale")
    parser.add_argument("-k", "--k_sparse_factor", type=int, default=10, help="k_sparse factor")
    parser.add_argument("-p", "--path", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("-i", "--n_iter", type=int, default=100, help="Number of iterations of MFACO to run")
    parser.add_argument("-n", "--n_ants", type=int, default=100, help="Number of ants")
    parser.add_argument("-r", "--n_runs", type=int, default=10, help="Number of runs per instance")
    parser.add_argument("-d", "--device", type=str,
                        default=("cuda:0" if torch.cuda.is_available() else "cpu"), 
                        help="The device to train NNs")
    ### GFACS
    parser.add_argument("--disable_guided_exp", action='store_true', help='True for model w/o guided exploration.')
    ### Seed
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("-g", "--gfacs", action='store_true', help="Loading GFACS model")
    parser.add_argument("--ppo", action='store_true', help="Loading PPO model")
    parser.add_argument("--mfaco", action='store_true', help="Use mfaco model")
    ### NumPy option
    parser.add_argument("--numpy", action='store_true', help="Use NumPy ACO implementation instead of PyTorch MFACO")
    args = parser.parse_args()

    DEVICE = args.device if torch.cuda.is_available() else 'cpu'
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
        args.k_sparse_factor,
        args.n_ants,
        args.n_iter,
        not args.disable_guided_exp,
        args.seed,
        args.n_runs,
        args.numpy
    )
