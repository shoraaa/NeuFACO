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


def train_instance(
        model,
        optimizer,
        data,
        n_ants,
        cost_w=1.0,
        invtemp=1.0,
        guided_exploration=False,
        shared_energy_norm=False,
        clip_ratio=0.2,
        value_coeff=0.5,
        entropy_coeff=0.01,
        it=0,
    ):
    model.train()

    ##################################################
    # wandb
    _train_mean_cost = 0.0
    _train_min_cost = 0.0
    _train_mean_cost_nls = 0.0
    _train_min_cost_nls = 0.0
    _train_entropy = 0.0
    _train_value_loss = 0.0
    _train_policy_loss = 0.0
    ##################################################
    sum_loss = torch.tensor(0.0, device=DEVICE)
    count = 0

    for pyg_data, distances in data:
        heu_vec, values = model(pyg_data, return_value=True)
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS

        aco = ACO(distances, n_ants, heuristic=heu_mat, device=DEVICE, local_search_type='2opt')

        # Sample paths and get old log probabilities
        costs, old_log_probs, paths = aco.sample(invtemp=invtemp, start_node=START_NODE)
        
        # Compute returns (negative costs as rewards)
        rewards = -costs
        if shared_energy_norm:
            rewards = rewards - rewards.mean()
        
        # Apply local search if guided exploration is enabled
        if guided_exploration:
            paths_nls = aco.local_search(paths, inference=False)
            costs_nls = aco.gen_path_costs(paths_nls)
            rewards_nls = -costs_nls
            if shared_energy_norm:
                rewards_nls = rewards_nls - rewards_nls.mean()
            
            # Weighted combination of rewards
            final_rewards = cost_w * rewards_nls + (1 - cost_w) * rewards
            
            # Get log probabilities for improved paths
            _, new_log_probs_nls = aco.gen_path(
                invtemp=invtemp,
                require_prob=True,
                paths=paths_nls,
                start_node=START_NODE,
            )
        else:
            final_rewards = rewards

        # Compute advantages using value function
        # The value function should output a single value per instance, not per ant
        predicted_values = values.squeeze(-1)  # Remove last dimension if needed
        if len(predicted_values.shape) > 1:
            predicted_values = predicted_values.mean(dim=0)  # Average over nodes if needed
        
        # Expand predicted values to match the number of ants
        predicted_values = predicted_values.expand(n_ants)
        
        advantages = final_rewards - predicted_values.detach()
        returns = final_rewards

        # PPO policy loss
        new_log_probs = old_log_probs  # For standard paths
        ratio = torch.exp(new_log_probs.sum(0) - old_log_probs.sum(0).detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value function loss
        value_loss = torch.nn.functional.mse_loss(predicted_values, returns.detach())

        # Entropy bonus for exploration
        normed_heumat = heu_mat / heu_mat.sum(dim=1, keepdim=True)
        entropy = -(normed_heumat * torch.log(normed_heumat + EPS)).sum(dim=1).mean()
        entropy_loss = -entropy_coeff * entropy

        # Additional loss for guided exploration paths
        if guided_exploration:
            ratio_nls = torch.exp(new_log_probs_nls.sum(0) - old_log_probs.sum(0).detach())
            surr1_nls = ratio_nls * advantages
            surr2_nls = torch.clamp(ratio_nls, 1 - clip_ratio, 1 + clip_ratio) * advantages
            policy_loss_nls = -torch.min(surr1_nls, surr2_nls).mean()
            policy_loss = (policy_loss + policy_loss_nls) / 2

        # Total loss
        total_loss = policy_loss + value_coeff * value_loss + entropy_loss
        sum_loss += total_loss

        count += 1

        ##################################################
        # wandb
        if USE_WANDB:
            _train_mean_cost += costs.mean().item()
            _train_min_cost += costs.min().item()
            _train_entropy += entropy.item()
            _train_value_loss += value_loss.item()
            _train_policy_loss += policy_loss.item()

            if guided_exploration:
                _train_mean_cost_nls += costs_nls.mean().item()
                _train_min_cost_nls += costs_nls.min().item()
        ##################################################

    sum_loss = sum_loss / count
    loss = sum_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=3.0, norm_type=2)  # type: ignore
    optimizer.step()

    ##################################################
    # wandb
    if USE_WANDB:
        wandb.log(
            {
                "train_mean_cost": _train_mean_cost / count,
                "train_min_cost": _train_min_cost / count,
                "train_mean_cost_nls": _train_mean_cost_nls / count,
                "train_min_cost_nls": _train_min_cost_nls / count,
                "train_entropy": _train_entropy / count,
                "train_policy_loss": _train_policy_loss / count,
                "train_value_loss": _train_value_loss / count,
                "train_total_loss": sum_loss.item(),
                "cost_w": cost_w,
                "invtemp": invtemp,
                "clip_ratio": clip_ratio,
                "value_coeff": value_coeff,
                "entropy_coeff": entropy_coeff,
            },
            step=it,
        )
    ##################################################


def infer_instance(model, pyg_data, distances, n_ants):
    model.eval()
    heu_vec = model(pyg_data)  # Only get heuristics, not value during inference
    heu_mat = model.reshape(pyg_data, heu_vec) + EPS

    aco = MFACO_NP(
        distances.cpu().numpy(),
        n_ants,
        heuristic=heu_mat.cpu().numpy(),
        local_search_type='2opt'
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
    cost_w=0.95,
    invtemp=1.0,
    guided_exploration=False,
    shared_energy_norm=False,
    clip_ratio=0.2,
    value_coeff=0.5,
    entropy_coeff=0.01,
):
    for i in tqdm(range(steps_per_epoch), desc="Train", dynamic_ncols=True):
        it = (epoch - 1) * steps_per_epoch + i
        data = generate_traindata(batch_size, n_node, k_sparse)
        train_instance(net, optimizer, data, n_ants, cost_w, invtemp, guided_exploration, shared_energy_norm, clip_ratio, value_coeff, entropy_coeff, it)


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
        savepath="../pretrained/tsp_ppo",
        run_name="",
        cost_w_schedule_params=(0.5, 1.0, 5),  # (cost_w_min, cost_w_max, cost_w_flat_epochs)
        invtemp_schedule_params=(0.8, 1.0, 5),  # (invtemp_min, invtemp_max, invtemp_flat_epochs)
        guided_exploration=False,
        shared_energy_norm=False,
        clip_ratio_schedule_params=(0.1, 0.2, 5),  # (clip_ratio_min, clip_ratio_max, clip_ratio_flat_epochs)
        value_coeff=0.5,
        entropy_coeff=0.01,
    ):
    savepath = os.path.join(savepath, str(n_nodes), run_name)
    os.makedirs(savepath, exist_ok=True)

    net = Net(gfn=False, value_head=True, start_node=START_NODE).to(DEVICE)  # PPO needs value head, not GFN
    if pretrained:
        net.load_state_dict(torch.load(pretrained, map_location=DEVICE))
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=lr * 0.1)

    val_list = load_val_dataset(n_nodes, k_sparse, DEVICE, start_node=START_NODE)
    val_list = val_list[:(val_size or len(val_list))]

    best_result = validation(val_list, n_val_ants, net, 0, steps_per_epoch)

    sum_time = 0
    for epoch in range(1, epochs + 1):
        # Cost Weight Schedule
        cost_w_min, cost_w_max, cost_w_flat_epochs = cost_w_schedule_params
        cost_w = cost_w_min + (cost_w_max - cost_w_min) * min((epoch - 1) / (epochs - cost_w_flat_epochs), 1.0)

        # Heatmap Inverse Temperature Schedule
        invtemp_min, invtemp_max, invtemp_flat_epochs = invtemp_schedule_params
        invtemp = invtemp_min + (invtemp_max - invtemp_min) * min((epoch - 1) / (epochs - invtemp_flat_epochs), 1.0)

        # Clip Ratio Schedule for PPO
        clip_ratio_min, clip_ratio_max, clip_ratio_flat_epochs = clip_ratio_schedule_params
        clip_ratio = clip_ratio_min + (clip_ratio_max - clip_ratio_min) * min((epoch - 1) / (epochs - clip_ratio_flat_epochs), 1.0)

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
            cost_w,
            invtemp,
            guided_exploration,
            shared_energy_norm,
            clip_ratio,
            value_coeff,
            entropy_coeff,
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
    parser.add_argument("-e", "--epochs", type=int, default=50, help="Epochs to run")
    parser.add_argument("-v", "--val_size", type=int, default=20, help="Number of instances for validation")
    parser.add_argument("-o", "--output", type=str, default="../pretrained/tsp_ppo",
                        help="The directory to store checkpoints")
    parser.add_argument("--val_interval", type=int, default=1, help="The interval to validate model")
    ### Logging
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--run_name", type=str, default="", help="Run name")
    ### invtemp
    parser.add_argument("--invtemp_min", type=float, default=1.0, help='Inverse temperature min')
    parser.add_argument("--invtemp_max", type=float, default=1.0, help='Inverse temperature max')
    parser.add_argument("--invtemp_flat_epochs", type=int, default=5, help='Inverse temperature flat epochs')
    ### PPO parameters
    parser.add_argument("--disable_guided_exp", action='store_true', help='Disable guided exploration')
    parser.add_argument("--disable_shared_energy_norm", action='store_true', help='Disable shared energy normalization')
    parser.add_argument("--clip_ratio_min", type=float, default=0.1, help='PPO clip ratio min')
    parser.add_argument("--clip_ratio_max", type=float, default=0.2, help='PPO clip ratio max')
    parser.add_argument("--clip_ratio_flat_epochs", type=int, default=5, help='PPO clip ratio flat epochs')
    parser.add_argument("--value_coeff", type=float, default=0.5, help='Value function loss coefficient')
    parser.add_argument("--entropy_coeff", type=float, default=0.01, help='Entropy loss coefficient')
    ### Energy Reshaping
    parser.add_argument("--cost_w_min", type=float, default=None, help='Cost weight min')
    parser.add_argument("--cost_w_max", type=float, default=0.99, help='Cost weight max')
    parser.add_argument("--cost_w_flat_epochs", type=int, default=5, help='Cost weight flat epochs')
    ### Seed
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    if args.k_sparse is None:
        args.k_sparse = args.nodes // 10

    if args.cost_w_min is None:
        args.cost_w_min = 0.5 if args.pretrained is None else 0.8

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
        args.pretrained.replace("../pretrained/tsp_ppo/", "").replace("/", "_").replace(".pt", "")
        if args.pretrained is not None else None
    )
    run_name += f"{'' if pretrained_name is None else '_fromckpt-'+pretrained_name}"
    if USE_WANDB:
        wandb.init(project="neufaco_data", name=run_name)
        wandb.config.update(args)
        wandb.config.update({"T": T, "model": "PPO"})
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
        cost_w_schedule_params=(args.cost_w_min, args.cost_w_max, args.cost_w_flat_epochs),
        invtemp_schedule_params=(args.invtemp_min, args.invtemp_max, args.invtemp_flat_epochs),
        guided_exploration=(not args.disable_guided_exp),
        shared_energy_norm=(not args.disable_shared_energy_norm),
        clip_ratio_schedule_params=(args.clip_ratio_min, args.clip_ratio_max, args.clip_ratio_flat_epochs),
        value_coeff=args.value_coeff,
        entropy_coeff=args.entropy_coeff,
    )