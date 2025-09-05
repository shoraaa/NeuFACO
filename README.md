# Neural Focused Ant Colony Optimization
Neural Focused Ant Colony Optimization (NeuFACO) is a non-autoregressive framework for solving the Traveling Salesman Problem (TSP). It combines reinforcement learning with advanced Ant Colony Optimization (ACO) techniques to achieve both efficiency and solution quality.

## Key Features

### Reinforcement Learning Integration:

Uses Proximal Policy Optimization (PPO) to train a Graph Neural Network (GNN).

Generates instance-specific heuristic guidance rather than relying on fixed training schemes.

###Enhanced ACO Framework:

Incorporates candidate lists, focused tour modification, and scalable local search.

Efficiently integrates the learned heuristic into the solution process.

### Performance:

Leverages amortized inference together with the parallel stochastic exploration of ACO.

Provides fast and high-quality solutions across diverse TSP instances.
## Dependencies
- Python 3.11.5
- PyTorch 2.1.1
- PyTorch Geometric 2.4.0

We strongly recommend using [uv](https://github.com/astral-sh/uv) for virtual environment in this project.

```
uv venv python=3.11
```

```
uv pip install torch torch_geometric
```

```
uv pip install numpy numba pandas pyvrp scipy tdqm wandb
```

For the complete list of dependencies, please refer to the `requirements.txt` file.

## Usage
For the usage of the code, please refer to each folder's `README.md` file.


