# TSP Algorithm Color Scheme Reference

This document describes the consistent color scheme used across all TSP iteration plots.

## Algorithm Colors

| Display Name | Raw Algorithm Names           | Color Code | Color Name | Visual |
|--------------|------------------------------|------------|------------|--------|
| DeepACO      | deepaco_200, deepaco_500     | #1f77b4   | Blue       | 🔵     |
| GFACS        | gfacs_200, gfacs_500         | #ff7f0e   | Orange     | 🟠     |
| NeuFACO      | ppo_faco_200, ppo_faco_500   | #2ca02c   | Green      | 🟢     |
| ACO          | none                         | #d62728   | Red        | 🔴     |

## Line Styles (for normalized comparison plot)

| TSP Size | Line Style | Description |
|----------|------------|-------------|
| 200      | -          | Solid line  |
| 500      | --         | Dashed line |
| 1000     | -.         | Dash-dot line |

## Plot Types Generated

1. **tsp_iterations_comparison_by_size.png**
   - Side-by-side comparison for each TSP size
   - Same algorithm = same color across all sizes
   - Shows smooth convergence curves + original data points

2. **tsp_iterations_normalized_comparison.png**
   - All algorithms and sizes on one plot
   - Same algorithm = same color, different sizes = different line styles
   - Labels show display names (e.g., "DeepACO_TSP200", "NeuFACO_TSP500")
   - Normalized to initial cost for fair comparison

3. **tsp_diversity_comparison_by_size.png**
   - Diversity evolution for each TSP size
   - Same color scheme as cost plots
   - Shows how solution diversity changes during optimization

## Color Consistency Rules

- **Same Algorithm Family**: All variants of an algorithm (e.g., deepaco_200, deepaco_500) use the same base color
- **Cross-Plot Consistency**: The same algorithm has the same color in all three plot types
- **Size Differentiation**: In the normalized plot, different TSP sizes use different line styles while maintaining color consistency