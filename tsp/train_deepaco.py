import math
import os
import random
import time

from tqdm import tqdm
import numpy as np
import torch

from net import Net
from aco import ACO, ACO_NP
from mfaco import MFACO, MFACO_NP
from utils import gen_pyg_data, load_val_dataset

import wandb


EPS = 1e-10
T = 50  # ACO iterations for validation
START_NODE = None  # GFACS uses node coords as model input and the start_node is randomly chosen.


def train_instance_reinforce(
        model,
        optimizer,
        data,
        n_ants,
        baseline_type='mean',  # 'mean', 'critic', or 'greedy'
        critic_model=None,
        it=0,
    ):
    """
    REINFORCE training algorithm for the TSP neural heuristic model.
    
    Args:
        model: Neural network model that outputs heuristics
        optimizer: Optimizer for the model
        data: Training data (pyg_data, distances) pairs
        n_ants: Number of ants for sampling
        baseline_type: Type of baseline ('mean', 'critic', 'greedy')
        critic_model: Critic network for baseline (if baseline_type='critic')
        it: Current iteration for logging
    """
    model.train()
    if critic_model is not None:
        critic_model.train()

    ##################################################
    # wandb tracking
    _train_mean_cost = 0.0
    _train_min_cost = 0.0
    _train_entropy = 0.0
    _train_baseline = 0.0
    ##################################################
    
    sum_loss = torch.tensor(0.0, device=DEVICE)
    count = 0

    for pyg_data, distances in data:
        # Forward pass through the neural network to get heuristics
        heu_vec = model(pyg_data)
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS
        
        # Create ACO solver with learned heuristics
        aco = ACO(distances, n_ants, heuristic=heu_mat, device=DEVICE, local_search_type=None)
        
        # Sample paths and get log probabilities
        costs, log_probs, paths = aco.sample(invtemp=1.0, inference=False, start_node=START_NODE)
        
        # Calculate rewards (negative costs for maximization)
        rewards = -costs
        
        # Calculate baseline for variance reduction
        if baseline_type == 'mean':
            baseline = rewards.mean()
        elif baseline_type == 'critic' and critic_model is not None:
            baseline = critic_model(pyg_data)[0]  # Use first output if critic returns multiple values
        elif baseline_type == 'greedy':
            # Use greedy rollout as baseline (single ant with greedy policy)
            greedy_aco = ACO(distances, 1, heuristic=heu_mat, device=DEVICE, local_search_type=None)
            greedy_costs, _, _ = greedy_aco.sample(invtemp=10.0, inference=False, start_node=START_NODE)
            baseline = -greedy_costs[0]
        else:
            baseline = 0.0
        
        # Calculate advantages
        advantages = rewards - baseline
        
        # REINFORCE loss: -log_prob * advantage
        # Sum log_probs over the sequence (each step of the tour)
        total_log_probs = log_probs.sum(0)  # Sum over sequence length, shape: (n_ants,)
        
        # Policy gradient loss
        policy_loss = -(total_log_probs * advantages.detach()).mean()
        
        sum_loss += policy_loss
        count += 1

        ##################################################
        # wandb logging
        if USE_WANDB:
            _train_mean_cost += costs.mean().item()
            _train_min_cost += costs.min().item()
            _train_baseline += (baseline.item() if isinstance(baseline, torch.Tensor) else baseline)
            
            # Calculate entropy for exploration tracking
            normed_heumat = heu_mat / heu_mat.sum(dim=1, keepdim=True)
            entropy = -(normed_heumat * torch.log(normed_heumat + EPS)).sum(dim=1).mean()
            _train_entropy += entropy.item()
        ##################################################

    # Average loss over all instances in the batch
    avg_loss = sum_loss / count
    
    # Backward pass and optimization
    optimizer.zero_grad()
    avg_loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0, norm_type=2)
    optimizer.step()

    ##################################################
    # wandb logging
    if USE_WANDB:
        wandb.log(
            {
                "train_mean_cost": _train_mean_cost / count,
                "train_min_cost": _train_min_cost / count,
                "train_baseline": _train_baseline / count,
                "train_entropy": _train_entropy / count,
                "train_loss": avg_loss.item(),
                "baseline_type": baseline_type,
            },
            step=it,
        )
    ##################################################


def infer_instance(model, pyg_data, distances, n_ants):
    model.eval()
    heu_vec = model(pyg_data)
    heu_mat = model.reshape(pyg_data, heu_vec) + EPS

    aco = MFACO_NP(
        distances.cpu().numpy(),
        n_ants,
        heuristic=heu_mat.cpu().numpy(),
        local_search_type='nls'
    )

    costs = aco.sample(inference=True, start_node=START_NODE)[0]
    baseline = costs.mean().item()
    best_sample_cost = costs.min().item()
    best_aco_1, diversity_1, _ = aco.run(n_iterations=1, start_node=START_NODE)
    best_aco_T, diversity_T, _ = aco.run(n_iterations=T - 1, start_node=START_NODE)
    return np.array([baseline, best_sample_cost, best_aco_1, best_aco_T, diversity_1, diversity_T])


def generate_traindata(count, n_node, k_sparse):
    for _ in range(count):
        instance = torch.rand(size=(n_node, 2), device=DEVICE)
        yield gen_pyg_data(instance, k_sparse, start_node=START_NODE)


def train_epoch(
    n_node,
    k_sparse,
    n_ants,
    epoch,
    steps_per_epoch,
    net,
    optimizer,
    batch_size,
    baseline_type='mean',
    critic_model=None,
):
    """
    Train one epoch using REINFORCE algorithm.
    """
    for i in tqdm(range(steps_per_epoch), desc="Train", dynamic_ncols=True):
        it = (epoch - 1) * steps_per_epoch + i
        data = generate_traindata(batch_size, n_node, k_sparse)
        train_instance_reinforce(
            net, 
            optimizer, 
            data, 
            n_ants, 
            baseline_type=baseline_type,
            critic_model=critic_model,
            it=it
        )


@torch.no_grad()
def validation(val_list, n_ants, net, epoch, steps_per_epoch):
    stats = []
    for data, distances in tqdm(val_list, desc="Val", dynamic_ncols=True):
        stats.append(infer_instance(net, data, distances, n_ants))
    avg_stats = [i.item() for i in np.stack(stats).mean(0)]

    ##################################################
    print(f"epoch {epoch}:", avg_stats)
    # wandb
    if USE_WANDB:
        wandb.log(
            {
                "val_baseline": avg_stats[0],
                "val_best_sample_cost": avg_stats[1],
                "val_best_aco_1": avg_stats[2],
                "val_best_aco_T": avg_stats[3],
                "val_diversity_1": avg_stats[4],
                "val_diversity_T": avg_stats[5],
                "epoch": epoch,
            },
            step=epoch * steps_per_epoch,
        )
    ##################################################

    return avg_stats[3]


def train(
        n_nodes,
        k_sparse,
        n_ants,
        n_val_ants,
        steps_per_epoch,
        epochs,
        lr=1e-4,
        batch_size=3,
        val_size=None,
        val_interval=5,
        pretrained=None,
        savepath="../pretrained/tsp_reinforce",
        run_name="",
        baseline_type='mean',  # 'mean', 'critic', or 'greedy'
        use_critic=False,
    ):
    """
    Train the TSP model using REINFORCE algorithm.
    
    Args:
        baseline_type: Type of baseline to use ('mean', 'critic', 'greedy')
        use_critic: Whether to use a critic network for baseline
    """
    savepath = os.path.join(savepath, str(n_nodes), run_name)
    os.makedirs(savepath, exist_ok=True)

    # Create the main policy network (no GFN components needed for REINFORCE)
    net = Net(gfn=False, Z_out_dim=1, start_node=START_NODE).to(DEVICE)
    if pretrained:
        net.load_state_dict(torch.load(pretrained, map_location=DEVICE))
    
    # Create critic network if needed
    critic_model = None
    if use_critic and baseline_type == 'critic':
        from net import Critic
        critic_model = Critic(start_node=START_NODE).to(DEVICE)
        # Combine parameters for joint optimization
        all_params = list(net.parameters()) + list(critic_model.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=lr)
    else:
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=lr * 0.1)

    val_list = load_val_dataset(n_nodes, k_sparse, DEVICE, start_node=START_NODE)
    val_list = val_list[:(val_size or len(val_list))]

    best_result = validation(val_list, n_val_ants, net, 0, steps_per_epoch)

    sum_time = 0
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_epoch(
            n_nodes,
            k_sparse,
            n_ants,
            epoch,
            steps_per_epoch,
            net,
            optimizer,
            batch_size,
            baseline_type=baseline_type,
            critic_model=critic_model,
        )
        sum_time += time.time() - start

        if epoch % val_interval == 0:
            curr_result = validation(val_list, n_val_ants, net, epoch, steps_per_epoch)
            if curr_result < best_result:
                torch.save(net.state_dict(), os.path.join(savepath, f"best.pt"))
                best_result = curr_result

            torch.save(net.state_dict(), os.path.join(savepath, f"{epoch}.pt"))

        scheduler.step()

    print('\ntotal training duration:', sum_time)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", metavar='N', type=int, help="Problem scale")
    parser.add_argument("-k", "--k_sparse", type=int, default=None, help="k_sparse")
    parser.add_argument("-l", "--lr", metavar='η', type=float, default=1e-3, help="Learning rate")
    parser.add_argument("-d", "--device", type=str,
                        default=("cuda:0" if torch.cuda.is_available() else "cpu"), 
                        help="The device to train NNs")
    parser.add_argument("-p", "--pretrained", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("-a", "--ants", type=int, default=30, help="Number of ants (in ACO algorithm)")
    parser.add_argument("-va", "--val_ants", type=int, default=50, help="Number of ants for validation")
    parser.add_argument("-b", "--batch_size", type=int, default=20, help="Batch size")
    parser.add_argument("-s", "--steps", type=int, default=20, help="Steps per epoch")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="Epochs to run")
    parser.add_argument("-v", "--val_size", type=int, default=20, help="Number of instances for validation")
    parser.add_argument("-o", "--output", type=str, default="../pretrained/tsp_reinforce",
                        help="The directory to store checkpoints")
    parser.add_argument("--val_interval", type=int, default=1, help="The interval to validate model")
    ### Logging
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--run_name", type=str, default="", help="Run name")
    ### REINFORCE-specific arguments
    parser.add_argument("--baseline_type", type=str, default="mean", 
                        choices=["mean", "critic", "greedy"],
                        help="Type of baseline for REINFORCE ('mean', 'critic', 'greedy')")
    parser.add_argument("--use_critic", action="store_true", 
                        help="Use critic network for baseline (only effective with --baseline_type critic)")
    ### Seed
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    if args.k_sparse is None:
        args.k_sparse = args.nodes // 10

    DEVICE = args.device if torch.cuda.is_available() else "cpu"
    USE_WANDB = not args.disable_wandb

    # seed everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ##################################################
    # wandb
    run_name = f"[{args.run_name}]" if args.run_name else ""
    run_name += f"tsp{args.nodes}_sd{args.seed}"
    pretrained_name = (
        args.pretrained.replace("../pretrained/", "").replace("/", "_").replace(".pt", "")
        if args.pretrained is not None else None
    )
    run_name += f"{'' if pretrained_name is None else '_fromckpt-'+pretrained_name}"
    if USE_WANDB:
        wandb.init(project="neufaco_data", name=run_name)
        wandb.config.update(args)
        wandb.config.update({"T": T, "model": "REINFORCE"})
    ##################################################

    train(
        args.nodes,
        args.k_sparse,
        args.ants,
        args.val_ants,
        args.steps,
        args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_size=args.val_size,
        val_interval=args.val_interval,
        pretrained=args.pretrained,
        savepath=args.output,
        run_name=run_name,
        baseline_type=args.baseline_type,
        use_critic=args.use_critic,
    )