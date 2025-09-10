# REINFORCE Training for TSP

This implementation provides a simplified REINFORCE algorithm for training neural heuristics for the Traveling Salesman Problem (TSP), replacing the complex GFACS (Generative Flow Ant Colony Search) approach with a more straightforward policy gradient method.

## Key Changes

### 1. Simplified Training Algorithm
- **Original**: Complex trajectory balance loss with forward/backward flows, guided exploration, and energy reshaping
- **REINFORCE**: Standard policy gradient with baseline subtraction for variance reduction

### 2. Core REINFORCE Implementation

The main training function `train_instance_reinforce()` implements:

```python
# Sample paths using neural heuristics
costs, log_probs, paths = aco.sample(invtemp=1.0, start_node=START_NODE)

# Calculate rewards (negative costs)
rewards = -costs

# Calculate baseline for variance reduction
baseline = calculate_baseline(rewards, baseline_type)

# Calculate advantages
advantages = rewards - baseline

# REINFORCE loss: -log_prob * advantage
total_log_probs = log_probs.sum(0)  # Sum over sequence
policy_loss = -(total_log_probs * advantages.detach()).mean()
```

### 3. Baseline Options

Three baseline types are supported:

1. **Mean Baseline** (`baseline_type='mean'`):
   - Uses the mean reward of current batch
   - Simple and stable
   - Good starting point

2. **Greedy Baseline** (`baseline_type='greedy'`):
   - Uses a greedy rollout (high temperature sampling)
   - More sophisticated baseline
   - Can provide better variance reduction

3. **Critic Baseline** (`baseline_type='critic'`):
   - Uses a separate critic network to estimate value
   - Most sophisticated approach
   - Requires training the critic jointly

### 4. Network Architecture

- **Simplified**: No GFN (Generative Flow Network) components
- **Standard**: Uses only the heuristic output from the neural network
- **Optional**: Critic network for value estimation (when using critic baseline)

## Usage

### Basic Training

```bash
# Train with mean baseline (simplest)
python train_deepaco.py 50 --baseline_type mean --epochs 20

# Train with greedy baseline (better variance reduction)
python train_deepaco.py 200 --baseline_type greedy --epochs 50

# Train with critic baseline (most sophisticated)
python train_deepaco.py 200 --baseline_type critic --use_critic --epochs 50
```

### Example Script

Run the provided example script to test the implementation:

```bash
python train_reinforce_example.py
```

This will train small TSP instances with different baseline types to demonstrate the approach.

### Command Line Arguments

Key arguments for REINFORCE training:

- `--baseline_type`: Choice of baseline ('mean', 'critic', 'greedy')
- `--use_critic`: Enable critic network (only effective with critic baseline)
- `--lr`: Learning rate (default: 1e-3)
- `--epochs`: Number of training epochs
- `--ants`: Number of ants for sampling during training
- `--batch_size`: Number of instances per training step

## Advantages of REINFORCE Approach

1. **Simplicity**: Much easier to understand and implement
2. **Stability**: More stable training without complex hyperparameter schedules
3. **Flexibility**: Easy to modify and extend
4. **Standard**: Uses well-established policy gradient methods
5. **Efficiency**: Fewer hyperparameters to tune

## Performance Considerations

- **Variance**: REINFORCE can have high variance; baseline helps reduce this
- **Sample Efficiency**: May require more samples than the original GFACS approach
- **Convergence**: Generally more stable convergence
- **Hyperparameters**: Fewer hyperparameters make it easier to tune

## Comparison with Original GFACS

| Aspect | GFACS | REINFORCE |
|--------|-------|-----------|
| Complexity | High | Low |
| Hyperparameters | Many (cost_w, beta, invtemp, etc.) | Few (baseline_type, lr) |
| Training Stability | Requires careful tuning | More stable |
| Implementation | Complex trajectory balance | Standard policy gradient |
| Flexibility | Specialized for TSP | General RL approach |

## Future Extensions

The REINFORCE implementation can be easily extended with:

1. **Advanced Baselines**: Self-critical training, exponential moving averages
2. **Entropy Regularization**: Add entropy bonus for exploration
3. **Multiple Objectives**: Handle multi-objective optimization
4. **Different Problems**: Adapt to other combinatorial problems (VRP, etc.)
5. **Actor-Critic**: Full actor-critic implementation with value function learning

## Tips for Best Results

1. Start with `baseline_type='mean'` for initial experiments
2. Use `baseline_type='greedy'` for better performance
3. Consider `baseline_type='critic'` for large-scale problems
4. Adjust learning rate based on problem size
5. Monitor entropy to ensure adequate exploration
6. Use gradient clipping (already implemented) for stability
