import os
import random

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import wandb

from net import Net
from aco import ACO
from utils import load_tsplib_dataset


EPS = 1e-10
START_NODE = None


@torch.no_grad()
def infer_instance(model, pyg_data, distances, n_ants, t_aco_diff, instance_name=None, use_wandb=True):
    if model is not None:
        model.eval()
        heu_vec = model(pyg_data)
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS
    else:
        heu_mat = None

    aco = ACO(
        distances.cpu(),
        n_ants,
        heuristic=heu_mat.cpu() if heu_mat is not None else heu_mat,
        device=DEVICE,
    )

    results = torch.zeros(size=(len(t_aco_diff),))
    elapsed_time = 0
    for i, t in enumerate(t_aco_diff):
        cost, _, t = aco.run(t, start_node=START_NODE)
        results[i] = cost
        elapsed_time += t
        print(f"Iteration {i+1}/{len(t_aco_diff)}: Cost = {cost}, Time = {t:.2f}s")
        
        # Log to wandb
        if use_wandb and instance_name is not None:
            wandb.log({
                f"cost/{instance_name}": cost,
                f"time/{instance_name}": t,
                "iteration": sum(t_aco_diff[:i+1]),
                "global_step": wandb.run.step if wandb.run else 0
            })
    return results, aco.shortest_path, elapsed_time


@torch.no_grad()
def test(dataset, scale_list, name_list, model, n_ants, t_aco, use_wandb=True):
    _t_aco = [0] + t_aco
    t_aco_diff = [_t_aco[i + 1] - _t_aco[i] for i in range(len(_t_aco) - 1)]

    results_list = []
    best_paths = []
    sum_times = 0
    for (pyg_data, distances), scale, name in tqdm(zip(dataset, scale_list, name_list)):
        ceiled_distances = (distances * scale).ceil()
        results, best_path, elapsed_time = infer_instance(model, pyg_data, ceiled_distances, n_ants, t_aco_diff, name, use_wandb)
        results_list.append(results)
        best_paths.append(best_path)
        sum_times += elapsed_time
    return results_list, best_paths, sum_times / len(dataset)


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


def main(ckpt_path, n_nodes, k_sparse_factor=10, n_ants=None, n_iter=10, guided_exploration=False, seed=0, use_wandb=True):
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
    print("seed:", seed)

    # Initialize wandb
    if use_wandb:
        model_name = os.path.splitext(os.path.basename(ckpt_path))[0] if ckpt_path is not None else 'no_model'
        wandb.init(
            project="tsplib-test",
            name=f"{model_name}-{n_nodes}nodes-{n_ants}ants-{n_iter}iter-seed{seed}",
            config={
                "n_nodes": n_nodes,
                "k_sparse_factor": k_sparse_factor,
                "n_ants": n_ants,
                "n_iter": n_iter,
                "guided_exploration": guided_exploration,
                "seed": seed,
                "checkpoint": ckpt_path,
                "model_type": "GFACS" if GFACS else "PPO",
                "device": DEVICE
            }
        )

    if ckpt_path is not None:
        # net = Net(gfn=True, Z_out_dim=2 if guided_exploration else 1, start_node=START_NODE).to(DEVICE) if GFACS else \
        #       Net(gfn=False, value_head=True, start_node=START_NODE).to(DEVICE) 
        net = Net(gfn=False, Z_out_dim=1, start_node=START_NODE).to(DEVICE)
        net.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    else:
        net = None
    results_list, best_paths, duration = test(test_list, scale_list, name_list, net, n_ants, t_aco, use_wandb)

    ### results_list is not consistent with the lengths calculated by below code. IDK why...
    ### Reload the original TSPlib data for cost calculation, as they rounds up the distances
    tsplib_data_list = [make_tsplib_data("../data/tsp/tsplib/TSPlib_70instances.txt", i) for i in range(70)]
    tsplib_instances = [(dat[0][0], dat[1], dat[2][0]) for dat in tsplib_data_list if dat[2][0] in name_list]
    assert len(tsplib_instances) == len(best_paths)

    results_df = pd.DataFrame(columns=["Length"])
    for inst, path in zip(tsplib_instances, best_paths):
        coords, _, tsp_name = inst
        tour_coords = np.concatenate([coords[path.cpu().numpy()], coords[path[0].cpu().numpy()].reshape(1, 2)], axis=0)
        tour_length = np.linalg.norm(tour_coords[1:] - tour_coords[:-1], axis=1)
        # round up to the nearest integer
        tour_length = np.ceil(tour_length).astype(int)
        tsp_results = np.sum(tour_length)
        results_df.loc[tsp_name, :] = tsp_results
        
        # Log final results to wandb
        if use_wandb:
            wandb.log({
                f"final_length/{tsp_name}": tsp_results,
            })

    print('average inference time: ', duration)
    
    # Log summary statistics
    if use_wandb:
        wandb.log({
            "average_inference_time": duration,
            "total_instances": len(tsplib_instances),
            "average_final_length": results_df["Length"].mean(),
            "min_final_length": results_df["Length"].min(),
            "max_final_length": results_df["Length"].max(),
        })

    # Save result in directory that contains model_file
    filename = os.path.splitext(os.path.basename(ckpt_path))[0] if ckpt_path is not None else 'none'
    dirname = os.path.dirname(ckpt_path) if ckpt_path is not None else f'../pretrained/tsp_ppo/{args.nodes}/no_model'
    os.makedirs(dirname, exist_ok=True)

    result_filename = f"test_result_ckpt{filename}-tsplib{n_nodes}-nants{n_ants}-niter{n_iter}-seed{seed}"
    results_df.to_csv(os.path.join(dirname, f"{result_filename}.csv"), index=True)
    
    # Finish wandb run
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", type=int, help="Problem scale")
    parser.add_argument("-k", "--k_sparse_factor", type=int, default=10, help="k_sparse factor")
    parser.add_argument("-p", "--path", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("-i", "--n_iter", type=int, default=50, help="Number of iterations of ACO to run")
    parser.add_argument("-n", "--n_ants", type=int, default=None, help="Number of ants")
    parser.add_argument("-d", "--device", type=str,
                        default=("cuda:0" if torch.cuda.is_available() else "cpu"), 
                        help="The device to train NNs")
    ### GFACS
    parser.add_argument("--disable_guided_exp", action='store_true', help='True for model w/o guided exploration.')
    ### Seed
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("-g", "--gfacs", action='store_true', help="Loading GFACS model")
    ### Wandb
    parser.add_argument("--no_wandb", action='store_true', help="Disable wandb logging")
    args = parser.parse_args()

    DEVICE = args.device if torch.cuda.is_available() else 'cpu'
    GFACS = args.gfacs

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
        not args.no_wandb
    )
