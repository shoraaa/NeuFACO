import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
from pathlib import Path
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d

# Set global font sizes for better readability
plt.rcParams.update({
    'font.size': 14,           # Default font size
    'axes.titlesize': 16,      # Title font size
    'axes.labelsize': 14,      # Axis label font size
    'xtick.labelsize': 12,     # X-axis tick label size
    'ytick.labelsize': 12,     # Y-axis tick label size
    'legend.fontsize': 12,     # Legend font size
    'figure.titlesize': 18     # Figure title font size
})

def smooth_curve(x, y, method='spline', window_size=5):
    """
    Smooth curve using different methods
    
    Args:
        x: x-axis values
        y: y-axis values
        method: 'spline', 'gaussian', or 'moving_average'
        window_size: for moving average or gaussian sigma
    
    Returns:
        x_smooth, y_smooth: smoothed coordinates
    """
    if len(x) < 4:  # Not enough points for smoothing
        return x, y
    
    if method == 'spline':
        # Create more points for smoother curve
        x_new = np.linspace(x.min(), x.max(), len(x) * 3)
        try:
            spl = make_interp_spline(x, y, k=3)  # cubic spline
            y_smooth = spl(x_new)
            return x_new, y_smooth
        except:
            # Fallback to original if spline fails
            return x, y
    
    elif method == 'gaussian':
        y_smooth = gaussian_filter1d(y, sigma=window_size/3)
        return x, y_smooth
    
    elif method == 'moving_average':
        # Simple moving average
        y_smooth = np.convolve(y, np.ones(window_size)/window_size, mode='same')
        return x, y_smooth
    
    return x, y

def parse_filename(filepath):
    """Extract algorithm name and TSP size from CSV filename"""
    filename = os.path.basename(filepath)
    
    # Extract algorithm name (between 'ckpt' and '-tsp')
    alg_match = re.search(r'ckpt([^-]+)-tsp', filename)
    algorithm = alg_match.group(1) if alg_match else 'unknown'
    
    # Extract TSP size (between 'tsp' and '-')
    size_match = re.search(r'tsp(\d+)', filename)
    tsp_size = int(size_match.group(1)) if size_match else 0
    
    return algorithm, tsp_size

def get_display_name(algorithm):
    """Convert raw algorithm names to proper display names"""
    name_mapping = {
        'deepaco_200': 'DeepACO',
        'deepaco_500': 'DeepACO',
        'gfacs_200': 'GFACS',
        'gfacs_500': 'GFACS',
        'ppo_faco_200': 'NeuFACO',
        'ppo_faco_500': 'NeuFACO',
        'none': 'ACO'
    }
    return name_mapping.get(algorithm, algorithm)

def load_and_plot_iterations():
    """Load all iteration CSV files and create comparison plots"""
    
    # Find all iteration CSV files
    csv_pattern = "tsp/results_test/**/*_iterations.csv"
    csv_files = glob.glob(csv_pattern, recursive=True)
    
    if not csv_files:
        print("No iteration CSV files found!")
        return
    
    # Group data by TSP size and algorithm
    data_by_size = {}
    
    for csv_file in csv_files:
        try:
            algorithm, tsp_size = parse_filename(csv_file)
            
            # Skip if we couldn't parse the filename properly
            if tsp_size == 0:
                print(f"Skipping file with unknown size: {csv_file}")
                continue
                
            df = pd.read_csv(csv_file)
            
            # Initialize size group if not exists
            if tsp_size not in data_by_size:
                data_by_size[tsp_size] = {}
            
            # Store data with algorithm name as key
            data_by_size[tsp_size][algorithm] = df
            
            print(f"Loaded {algorithm} for TSP{tsp_size}: {len(df)} iterations")
            
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue
    
    # Create plots for each TSP size
    tsp_sizes = sorted(data_by_size.keys())
    
    # Set up consistent color palette for algorithms
    algorithms = set()
    for size_data in data_by_size.values():
        algorithms.update(size_data.keys())
    algorithms = sorted(list(algorithms))
    
    # Define consistent colors for display names
    display_name_colors = {
        'DeepACO': '#1f77b4',   # blue
        'GFACS': '#ff7f0e',     # orange
        'NeuFACO': '#2ca02c',   # green
        'ACO': '#d62728',       # red
    }
    
    # Fallback colors for any unknown algorithms
    fallback_colors = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Create color map based on display names
    color_map = {}
    fallback_idx = 0
    for alg in algorithms:
        display_name = get_display_name(alg)
        if display_name in display_name_colors:
            color_map[alg] = display_name_colors[display_name]
        else:
            color_map[alg] = fallback_colors[fallback_idx % len(fallback_colors)]
            fallback_idx += 1
    
    # Create subplots for each TSP size
    fig, axes = plt.subplots(1, len(tsp_sizes), figsize=(8*len(tsp_sizes), 7))
    if len(tsp_sizes) == 1:
        axes = [axes]
    
    for idx, tsp_size in enumerate(tsp_sizes):
        ax = axes[idx]
        
        for algorithm in sorted(data_by_size[tsp_size].keys()):
            df = data_by_size[tsp_size][algorithm]
            
            # Smooth the curves
            x_smooth, y_smooth = smooth_curve(df['T'].values, df['avg_cost'].values, method='spline')
            
            # Plot average cost over iterations (smooth)
            ax.plot(x_smooth, y_smooth, 
                   label=get_display_name(algorithm), 
                   color=color_map[algorithm], 
                   linewidth=2.5, 
                   alpha=0.9)
            
            # Plot original data points (optional, smaller and more transparent)
            ax.scatter(df['T'], df['avg_cost'], 
                      color=color_map[algorithm], 
                      s=15, 
                      alpha=0.4, 
                      zorder=5)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Average Cost')
        ax.set_title(f'TSP{tsp_size}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tsp_iterations_comparison_by_size.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create normalized comparison plot (all sizes together)
    plt.figure(figsize=(8, 6))
    
    # Define line styles for different TSP sizes
    size_styles = {200: '-', 500: '--', 1000: '-.'}
    
    for tsp_size in tsp_sizes:
        for algorithm in sorted(data_by_size[tsp_size].keys()):
            df = data_by_size[tsp_size][algorithm]
            
            # Normalize costs for better comparison across sizes
            initial_cost = df['avg_cost'].iloc[0]
            normalized_cost = df['avg_cost'] / initial_cost
            
            # Smooth the normalized curves
            x_smooth, y_smooth = smooth_curve(df['T'].values, normalized_cost.values, method='spline')
            
            # Create consistent label and use consistent color + line style
            display_name = get_display_name(algorithm)
            label = f"{display_name}_TSP{tsp_size}"
            linestyle = size_styles.get(tsp_size, '-')
            
            plt.plot(x_smooth, y_smooth, 
                    label=label, 
                    color=color_map[algorithm],  # Consistent color per algorithm
                    linestyle=linestyle,         # Different line style per size
                    linewidth=2.5, 
                    alpha=0.8)
    
    plt.xlabel('Iteration')
    plt.ylabel('Normalized Average Cost (relative to initial)')
    plt.title('Normalized Convergence')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tsp_iterations_normalized_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create diversity plots
    fig, axes = plt.subplots(1, len(tsp_sizes), figsize=(8*len(tsp_sizes), 7))
    if len(tsp_sizes) == 1:
        axes = [axes]
    
    for idx, tsp_size in enumerate(tsp_sizes):
        ax = axes[idx]
        
        for algorithm in sorted(data_by_size[tsp_size].keys()):
            df = data_by_size[tsp_size][algorithm]
            
            # Smooth the diversity curves
            x_smooth, y_smooth = smooth_curve(df['T'].values, df['avg_diversity'].values, method='spline')
            
            # Plot diversity over iterations (smooth)
            ax.plot(x_smooth, y_smooth, 
                   label=get_display_name(algorithm), 
                   color=color_map[algorithm], 
                   linewidth=2.5, 
                   alpha=0.9)
            
            # Plot original data points (optional, smaller and more transparent)
            ax.scatter(df['T'], df['avg_diversity'], 
                      color=color_map[algorithm], 
                      s=15, 
                      alpha=0.4, 
                      marker='s',
                      zorder=5)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Average Diversity')
        ax.set_title(f'TSP{tsp_size}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tsp_diversity_comparison_by_size.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for tsp_size in tsp_sizes:
        print(f"\nTSP Size: {tsp_size}")
        print("-" * 30)
        
        for algorithm in sorted(data_by_size[tsp_size].keys()):
            df = data_by_size[tsp_size][algorithm]
            initial_cost = df['avg_cost'].iloc[0]
            final_cost = df['avg_cost'].iloc[-1]
            improvement = ((initial_cost - final_cost) / initial_cost) * 100
            
            display_name = get_display_name(algorithm)
            print(f"{display_name:15s}: Initial={initial_cost:.3f}, "
                  f"Final={final_cost:.3f}, Improvement={improvement:.2f}%")

if __name__ == "__main__":
    load_and_plot_iterations()