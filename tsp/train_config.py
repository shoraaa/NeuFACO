#!/usr/bin/env python3
"""
Configuration-based training script for easier hyperparameter tuning.
"""

import yaml
import json
import argparse
from train import train, DEVICE, USE_WANDB
import torch
import numpy as np
import random
import wandb

# Default configuration
DEFAULT_CONFIG = {
    # Problem setup
    'nodes': 200,
    'k_sparse': None,  # Will be set to nodes // 10 if None
    
    # Training parameters
    'lr': 1e-3,
    'batch_size': 20,
    'steps': 20,
    'epochs': 50,
    'val_size': 20,
    'val_interval': 5,
    
    # ACO parameters
    'ants': 30,
    'val_ants': 50,
    
    # PPO parameters
    'clip_ratio_min': 0.1,
    'clip_ratio_max': 0.2,
    'clip_ratio_flat_epochs': 5,
    'value_coeff': 0.5,
    'entropy_coeff': 0.01,
    
    # Temperature scheduling
    'invtemp_min': 1.0,
    'invtemp_max': 1.0,
    'invtemp_flat_epochs': 5,
    
    # Cost weighting
    'cost_w_min': 0.5,
    'cost_w_max': 0.99,
    'cost_w_flat_epochs': 5,
    
    # Flags
    'guided_exploration': True,
    'shared_energy_norm': True,
    
    # I/O
    'pretrained': None,
    'savepath': '../pretrained/tsp_ppo',
    'run_name': '',
    'disable_wandb': False,
    'seed': 0,
    'device': 'cuda:0'
}


def load_config(config_path):
    """Load configuration from YAML or JSON file."""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    
    # Merge with defaults
    final_config = DEFAULT_CONFIG.copy()
    final_config.update(config)
    return final_config


def train_with_config(config):
    """Train model with given configuration."""
    
    # Set k_sparse if not provided
    if config['k_sparse'] is None:
        config['k_sparse'] = config['nodes'] // 10
    
    # Set device
    global DEVICE
    DEVICE = config['device'] if torch.cuda.is_available() else "cpu"
    
    # Set wandb usage
    global USE_WANDB
    USE_WANDB = not config['disable_wandb']
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    
    # Initialize wandb if enabled
    if USE_WANDB:
        run_name = f"{config['run_name']}_" if config['run_name'] else ""
        run_name += f"tsp{config['nodes']}_sd{config['seed']}"
        
        wandb.init(
            project="neufaco-tuning",
            name=run_name,
            config=config
        )
    
    # Call training function
    result = train(
        n_nodes=config['nodes'],
        k_sparse=config['k_sparse'],
        n_ants=config['ants'],
        n_val_ants=config['val_ants'],
        steps_per_epoch=config['steps'],
        epochs=config['epochs'],
        lr=config['lr'],
        batch_size=config['batch_size'],
        val_size=config['val_size'],
        val_interval=config['val_interval'],
        pretrained=config['pretrained'],
        savepath=config['savepath'],
        run_name=run_name if USE_WANDB else config['run_name'],
        cost_w_schedule_params=(
            config['cost_w_min'],
            config['cost_w_max'],
            config['cost_w_flat_epochs']
        ),
        invtemp_schedule_params=(
            config['invtemp_min'],
            config['invtemp_max'],
            config['invtemp_flat_epochs']
        ),
        guided_exploration=config['guided_exploration'],
        shared_energy_norm=config['shared_energy_norm'],
        clip_ratio_schedule_params=(
            config['clip_ratio_min'],
            config['clip_ratio_max'],
            config['clip_ratio_flat_epochs']
        ),
        value_coeff=config['value_coeff'],
        entropy_coeff=config['entropy_coeff'],
    )
    
    if USE_WANDB:
        wandb.finish()
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Config-based training for PPO-TSP')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file (YAML or JSON)')
    parser.add_argument('--override', type=str, nargs='*', default=[],
                       help='Override config values (e.g., --override lr=0.001 batch_size=32)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply overrides
    for override in args.override:
        key, value = override.split('=')
        # Try to convert to appropriate type
        try:
            if '.' in value:
                value = float(value)
            elif value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
        except:
            pass  # Keep as string
        
        config[key] = value
    
    print("Training with configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Run training
    train_with_config(config)


if __name__ == '__main__':
    main()
