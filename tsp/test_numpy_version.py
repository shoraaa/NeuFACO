#!/usr/bin/env python3
"""
Test script to verify the NumPy version of the ACO implementation works correctly.
"""

import numpy as np
import torch
from test_tsplib_m import infer_instance_np, infer_instance
from aco import ACO_NP
from mfaco import MFACO, MFACO_NP

def test_numpy_vs_torch():
    """Compare NumPy and PyTorch implementations on a simple example."""
    
    # Create a simple distance matrix for testing
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_nodes = 20
    coords = np.random.rand(n_nodes, 2) * 100
    
    # Calculate distance matrix
    distances_np = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    distances_torch = torch.from_numpy(distances_np).float()
    
    # Test parameters
    n_ants = 32
    t_aco_diff = [10, 10, 10]  # 3 iterations of 10 each
    n_runs = 2
    
    print("Testing NumPy vs PyTorch ACO implementations...")
    print(f"Problem size: {n_nodes} nodes")
    print(f"Number of ants: {n_ants}")
    print(f"Iterations: {t_aco_diff}")
    print(f"Number of runs: {n_runs}")
    print()
    
    # Test NumPy version (without model)
    print("Running NumPy version...")
    results_np, path_np, time_np, all_runs_np = infer_instance_np(
        model=None, 
        pyg_data=None, 
        distances=distances_np, 
        n_ants=n_ants, 
        t_aco_diff=t_aco_diff, 
        n_runs=n_runs
    )
    print(f"NumPy results: {results_np}")
    print(f"NumPy best cost: {results_np[-1]:.2f}")
    print(f"NumPy time: {time_np:.2f}s")
    print()
    
    # Test PyTorch version (without model)
    print("Running PyTorch version...")
    results_torch, path_torch, time_torch, all_runs_torch = infer_instance(
        model=None, 
        pyg_data=None, 
        distances=distances_torch, 
        n_ants=n_ants, 
        t_aco_diff=t_aco_diff, 
        n_runs=n_runs
    )
    print(f"PyTorch results: {results_torch}")
    print(f"PyTorch best cost: {results_torch[-1]:.2f}")
    print(f"PyTorch time: {time_torch:.2f}s")
    print()
    
    # Compare results
    print("Comparison:")
    print(f"Results difference (should be similar): {np.abs(results_np - results_torch.cpu().numpy()).max():.4f}")
    print(f"Time ratio (NumPy/PyTorch): {time_np/time_torch:.2f}")
    
    return results_np, results_torch, path_np, path_torch

def test_direct_aco():
    """Test ACO_NP directly to ensure it works."""
    
    print("\n" + "="*50)
    print("Testing ACO_NP directly...")
    
    # Create a simple distance matrix
    np.random.seed(42)
    n_nodes = 10
    coords = np.random.rand(n_nodes, 2) * 100
    distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    
    # Initialize ACO
    aco = ACO_NP(distances, n_ants=20)
    
    # Run ACO
    print(f"Running ACO on {n_nodes} nodes with 20 ants for 50 iterations...")
    cost, diversity, time_taken = aco.run(50)
    
    print(f"Best cost: {cost:.2f}")
    print(f"Diversity: {diversity:.4f}")
    print(f"Time taken: {time_taken:.2f}s")
    print(f"Best path length: {len(aco.shortest_path)}")
    
    return cost, aco.shortest_path

if __name__ == "__main__":
    # Test direct ACO implementation
    cost, path = test_direct_aco()
    
    # Test NumPy vs PyTorch versions
    try:
        results_np, results_torch, path_np, path_torch = test_numpy_vs_torch()
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
