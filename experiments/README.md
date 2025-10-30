This directory contains training scripts for all our experiments, including our main approach SLAP, baselines, and ablations.

| Approach | Uses Task Planning | Uses Abstractions | Training Method | Policy Type |
|----------|-------------------|-------------------|-----------------|-------------|
| SLAP | ✓ | ✓ | Multiple specialized policies | Shortcut-specific |
| Abstract Subgoals | ✓ | ✓ | Single context-conditioned policy | Goal-conditioned |
| Abstract HER | ✓ | ✓ | Single policy with HER | Goal-conditioned |
| Hierarchical RL | ✗ | Partial (skills) | Single skill-level policy | Hierarchical |
| Pure RL (PPO) | ✗ | ✗ | Single policy | Monolithic |
| SAC+HER | ✗ | ✗ | Single policy with HER | Monolithic |
| Pure Planning | ✓ | ✓ | No learning | Classical TAMP |