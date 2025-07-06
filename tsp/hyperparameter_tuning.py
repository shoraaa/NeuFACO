#!/usr/bin/env python3
"""
Hyperparameter tuning script for PPO-TSP training.
This script can be used with wandb sweeps or standalone optimization.
"""

import wandb
import argparse
import subprocess
import sys
import os
from itertools import product
import random

def create_sweep_config(problem_size=200):
    """Create a sweep configuration for wandb."""
    sweep_config = {
        'program': 'train.py',
        'method': 'bayes',  # 'random', 'grid', 'bayes'
        'metric': {
            'name': 'val_best_aco_T',
            'goal': 'minimize'
        },
        'parameters': {
            # Fixed parameters
            'nodes': {'value': problem_size},
            'epochs': {'value': 50},
            'val_interval': {'value': 5},
            'disable_wandb': {'value': False},
            
            # Core training parameters to tune
            'lr': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-2
            },
            'batch_size': {
                'values': [10, 20, 30, 50]
            },
            'steps': {
                'values': [10, 20, 30, 40]
            },
            
            # PPO parameters
            'clip_ratio_min': {
                'distribution': 'uniform',
                'min': 0.05,
                'max': 0.2
            },
            'clip_ratio_max': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.3
            },
            'value_coeff': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 1.0
            },
            'entropy_coeff': {
                'distribution': 'log_uniform_values',
                'min': 0.001,
                'max': 0.1
            },
            
            # ACO parameters
            'ants': {
                'values': [20, 30, 50, 80]
            },
            'invtemp_min': {
                'distribution': 'uniform',
                'min': 0.5,
                'max': 1.0
            },
            'invtemp_max': {
                'distribution': 'uniform',
                'min': 1.0,
                'max': 2.0
            },
            
            # Cost weighting
            'cost_w_min': {
                'distribution': 'uniform',
                'min': 0.3,
                'max': 0.7
            },
            'cost_w_max': {
                'distribution': 'uniform',
                'min': 0.8,
                'max': 0.99
            },
            
            # Boolean flags
            'disable_guided_exp': {
                'values': [True, False]
            },
            'disable_shared_energy_norm': {
                'values': [True, False]
            }
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 10
        }
    }
    return sweep_config


def grid_search_manual(problem_size=200, max_trials=50):
    """Manual grid search implementation."""
    
    # Define parameter grids
    param_grid = {
        'lr': [1e-4, 3e-4, 1e-3, 3e-3],
        'batch_size': [10, 20, 30],
        'steps': [10, 20, 30],
        'clip_ratio_min': [0.05, 0.1, 0.15],
        'clip_ratio_max': [0.15, 0.2, 0.25],
        'value_coeff': [0.3, 0.5, 0.7],
        'entropy_coeff': [0.01, 0.03, 0.05],
        'ants': [20, 30, 50],
        'disable_guided_exp': [True, False],
        'disable_shared_energy_norm': [True, False]
    }
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = list(product(*values))
    
    # Randomly sample combinations if too many
    if len(all_combinations) > max_trials:
        combinations = random.sample(all_combinations, max_trials)
    else:
        combinations = all_combinations
    
    best_config = None
    best_score = float('inf')
    
    for i, combination in enumerate(combinations):
        config = dict(zip(keys, combination))
        print(f"\nTrial {i+1}/{len(combinations)}")
        print(f"Config: {config}")
        
        # Build command
        cmd = [
            'python', 'train.py', str(problem_size),
            '--lr', str(config['lr']),
            '--batch_size', str(config['batch_size']),
            '--steps', str(config['steps']),
            '--ants', str(config['ants']),
            '--clip_ratio_min', str(config['clip_ratio_min']),
            '--clip_ratio_max', str(config['clip_ratio_max']),
            '--value_coeff', str(config['value_coeff']),
            '--entropy_coeff', str(config['entropy_coeff']),
            '--epochs', '25',  # Shorter for tuning
            '--run_name', f'grid_search_trial_{i+1}'
        ]
        
        if config['disable_guided_exp']:
            cmd.append('--disable_guided_exp')
        if config['disable_shared_energy_norm']:
            cmd.append('--disable_shared_energy_norm')
        
        try:
            # Run training
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                # Parse the output to get the final validation score
                # This is simplified - you might want to save results to a file
                print(f"Trial {i+1} completed successfully")
            else:
                print(f"Trial {i+1} failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"Trial {i+1} timed out")
        except Exception as e:
            print(f"Trial {i+1} error: {e}")
    
    return best_config, best_score


def random_search(problem_size=200, num_trials=30):
    """Random search implementation."""
    
    import random
    
    def sample_config():
        return {
            'lr': random.uniform(1e-5, 1e-2),
            'batch_size': random.choice([10, 20, 30, 50]),
            'steps': random.choice([10, 20, 30]),
            'clip_ratio_min': random.uniform(0.05, 0.2),
            'clip_ratio_max': random.uniform(0.1, 0.3),
            'value_coeff': random.uniform(0.1, 1.0),
            'entropy_coeff': random.uniform(0.001, 0.1),
            'ants': random.choice([20, 30, 50]),
            'invtemp_min': random.uniform(0.5, 1.0),
            'invtemp_max': random.uniform(1.0, 2.0),
            'cost_w_min': random.uniform(0.3, 0.7),
            'cost_w_max': random.uniform(0.8, 0.99),
            'disable_guided_exp': random.choice([True, False]),
            'disable_shared_energy_norm': random.choice([True, False])
        }
    
    best_config = None
    best_score = float('inf')
    
    for trial in range(num_trials):
        config = sample_config()
        print(f"\nTrial {trial+1}/{num_trials}")
        print(f"Config: {config}")
        
        # Build and run command similar to grid_search_manual
        # ... (implementation similar to above)
    
    return best_config, best_score


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for PPO-TSP')
    parser.add_argument('--method', choices=['wandb', 'grid', 'random'], default='wandb',
                       help='Tuning method to use')
    parser.add_argument('--problem_size', type=int, default=200,
                       help='TSP problem size')
    parser.add_argument('--max_trials', type=int, default=50,
                       help='Maximum number of trials')
    parser.add_argument('--project', type=str, default='neufaco-tuning',
                       help='Wandb project name')
    
    args = parser.parse_args()
    
    if args.method == 'wandb':
        # Initialize wandb sweep
        sweep_config = create_sweep_config(args.problem_size)
        sweep_id = wandb.sweep(sweep_config, project=args.project)
        
        print(f"Created sweep: {sweep_id}")
        print(f"Run with: wandb agent {sweep_id}")
        
    elif args.method == 'grid':
        print("Running manual grid search...")
        best_config, best_score = grid_search_manual(args.problem_size, args.max_trials)
        print(f"Best config: {best_config}")
        print(f"Best score: {best_score}")
        
    elif args.method == 'random':
        print("Running random search...")
        best_config, best_score = random_search(args.problem_size, args.max_trials)
        print(f"Best config: {best_config}")
        print(f"Best score: {best_score}")


if __name__ == '__main__':
    main()
