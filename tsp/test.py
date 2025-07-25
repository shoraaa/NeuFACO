import os
import random

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from math import sqrt

from net import Net
from mfaco import ACO
from utils import load_test_dataset


EPS = 1e-10
START_NODE = None


@torch.no_grad()
def infer_instance(model, pyg_data, distances, n_ants, t_aco_diff, k_sparse):
    heu_mat = None
    if model is not None:
        model.eval()
        heu_vec = model(pyg_data)
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS
        heu_mat = heu_mat.cpu()

    aco = ACO(
        distances.cpu(),
        n_ants,
        heuristic=heu_mat,
        k_sparse=k_sparse,
        device=DEVICE,
        local_search_type=None,
        elitist=ACOALG == "ELITIST",
        maxmin=ACOALG == "MAXMIN",
        rank_based=ACOALG == "RANK",
    )

    results = torch.zeros(size=(len(t_aco_diff),))
    diversities = torch.zeros(size=(len(t_aco_diff),))
    elapsed_time = 0
    for i, t in enumerate(t_aco_diff):
        results[i], diversities[i], t = aco.run(t, start_node=START_NODE)
        elapsed_time += t
    return results, diversities, elapsed_time


@torch.no_grad()
def test(dataset, model, n_ants, t_aco, k_sparse):
    _t_aco = [0] + t_aco
    t_aco_diff = [_t_aco[i + 1] - _t_aco[i] for i in range(len(_t_aco) - 1)]

    sum_results = torch.zeros(size=(len(t_aco_diff), ))
    sum_diversities = torch.zeros(size=(len(t_aco_diff), ))
    sum_times = 0
    for pyg_data, distances in tqdm(dataset, dynamic_ncols=True):
        results, diversities, elapsed_time = infer_instance(model, pyg_data, distances, n_ants, t_aco_diff, k_sparse)
        sum_results += results
        sum_diversities += diversities
        sum_times += elapsed_time
    return sum_results / len(dataset), sum_diversities / len(dataset), sum_times / len(dataset)


def main(ckpt_path, n_nodes, k_sparse, size=None, n_ants=None, n_iter=1000, guided_exploration=False, seed=0, test_name=""):
    test_list = load_test_dataset(n_nodes, k_sparse, DEVICE, start_node=START_NODE)
    test_list = test_list[:(size or len(test_list))]

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

    if ckpt_path is not None:
        net = Net(gfn=True, Z_out_dim=2 if guided_exploration else 1, start_node=START_NODE).to(DEVICE) if GFACS else \
              Net(gfn=False, value_head=True, start_node=START_NODE).to(DEVICE) 
        net.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    else:
        net = None
    avg_cost, avg_diversity, duration = test(test_list, net, n_ants, t_aco, k_sparse)
    print('average inference time: ', duration)
    for i, t in enumerate(t_aco):
        print(f"T={t}, avg. cost {avg_cost[i]}, avg. diversity {avg_diversity[i]}")

    # Save result in directory that contains model_file
    filename = os.path.splitext(os.path.basename(ckpt_path))[0] if ckpt_path is not None else 'none'
    dirname = os.path.dirname(ckpt_path) if ckpt_path is not None else f'../pretrained/tsp_nls/{args.nodes}/no_model'
    os.makedirs(dirname, exist_ok=True)

    result_filename = f"test_result_ckpt{filename}-tsp{n_nodes}-ninst{size}-{ACOALG}-nants{n_ants}-niter{n_iter}-seed{seed}{'-'+test_name if test_name else ''}"
    result_file = os.path.join(dirname, result_filename + ".txt")
    with open(result_file, "w") as f:
        f.write(f"problem scale: {n_nodes}\n")
        f.write(f"checkpoint: {ckpt_path}\n")
        f.write(f"number of instances: {len(test_list)}\n")
        f.write(f"device: {'cpu' if DEVICE == 'cpu' else DEVICE+'+cpu'}\n")
        f.write(f"n_ants: {n_ants}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"average inference time: {duration}\n")
        for i, t in enumerate(t_aco):
            f.write(f"T={t}, avg. cost {avg_cost[i]}, avg. diversity {avg_diversity[i]}\n")

    results = pd.DataFrame(columns=['T', 'avg_cost', 'avg_diversity'])
    results['T'] = t_aco
    results['avg_cost'] = avg_cost
    results['avg_diversity'] = avg_diversity
    results.to_csv(os.path.join(dirname, result_filename + ".csv"), index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", type=int, help="Problem scale")
    parser.add_argument("-k", "--k_sparse", type=int, default=None, help="k_sparse")
    parser.add_argument("-p", "--path", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("-i", "--n_iter", type=int, default=10, help="Number of iterations of ACO to run")
    parser.add_argument("-n", "--n_ants", type=int, default=None, help="Number of ants")
    parser.add_argument("-d", "--device", type=str,
                        default=("cuda:0" if torch.cuda.is_available() else "cpu"), 
                        help="The device to train NNs")
    parser.add_argument("-s", "--size", type=int, default=None, help="Number of instances to test")
    parser.add_argument("--test_name", type=str, default="", help="Name of the test")
    ### GFACS
    parser.add_argument("--disable_guided_exp", action='store_true', help='True for model w/o guided exploration.')
    ### Seed
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    ### ACO
    parser.add_argument("--aco", type=str, default="AS", choices=["AS", "ELITIST", "MAXMIN", "RANK"], help="ACO algorithm")
    parser.add_argument("-g", "--gfacs", action='store_true', help="Loading GFACS model")
    args = parser.parse_args()

    if args.k_sparse is None:
        args.k_sparse = args.nodes // 10

    DEVICE = args.device if torch.cuda.is_available() else 'cpu'
    ACOALG = args.aco
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
        args.k_sparse,
        args.size,
        args.n_ants,
        args.n_iter,
        not args.disable_guided_exp,
        args.seed,
        args.test_name
    )
