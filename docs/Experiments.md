# POCO-TTT Experiments

## Overview

POCO-TTT implements Test-Time Training (TTT) for neural activity prediction. This document describes the three main training methods being compared and their experimental setup.

## Methods Comparison

### Architecture Visualization

```
                    POCO-TTT Architecture Comparison
                    ================================

     POCO Baseline              FOMAML                    E2E-TTT
     (Standard Training)        (First-Order MAML)        (Second-Order Meta-Learning)

     ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
     │  Input: Neural  │       │  Input: Neural  │       │  Input: Neural  │
     │   Activity X    │       │   Activity X    │       │   Activity X    │
     └────────┬────────┘       └────────┬────────┘       └────────┬────────┘
              │                         │                         │
              ▼                         ▼                         ▼
     ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
     │  Unit Embedding │       │  Unit Embedding │       │  Unit Embedding │
     │   (Trainable)   │       │    (Adapted)    │       │    (Adapted)    │
     │                 │       │   Inner Loop    │       │   Inner Loop    │
     └────────┬────────┘       └────────┬────────┘       └────────┬────────┘
              │                         │                         │
              ▼                         ▼                         ▼
     ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
     │    Session      │       │    Session      │       │    Session      │
     │   Embedding     │       │   Embedding     │       │   Embedding     │
     │   (Trainable)   │       │    (Adapted)    │       │    (Adapted)    │
     └────────┬────────┘       └────────┬────────┘       └────────┬────────┘
              │                         │                         │
              ▼                         ▼                         ▼
     ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
     │   Perceiver-IO  │       │   Perceiver-IO  │       │   Perceiver-IO  │
     │    Backbone     │       │    Backbone     │       │    Backbone     │
     │   (Trainable)   │       │  (Meta-Trained) │       │  (Meta-Trained) │
     └────────┬────────┘       └────────┬────────┘       └────────┬────────┘
              │                         │                         │
              ▼                         ▼                         ▼
     ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
     │  Output: X̂      │       │  Output: X̂      │       │  Output: X̂      │
     │  (Prediction)   │       │  (Prediction)   │       │  (Prediction)   │
     └─────────────────┘       └─────────────────┘       └─────────────────┘
```

### Training Process Comparison

```
POCO Baseline Training:
=======================

For each batch:
    ┌─────────────────────────────────────────────────────────────┐
    │  1. Sample batch from ALL sessions (mixed)                   │
    │                                                              │
    │  2. Forward pass: X → model → X̂                             │
    │                                                              │
    │  3. Compute loss: MSE(X̂, X_target)                          │
    │                                                              │
    │  4. Backward pass: Update ALL parameters                     │
    │     (embeddings + backbone)                                  │
    └─────────────────────────────────────────────────────────────┘


FOMAML Training (Bilevel Optimization):
=======================================

For each meta-batch:
    ┌─────────────────────────────────────────────────────────────┐
    │  OUTER LOOP (Meta-Update):                                   │
    │  ┌─────────────────────────────────────────────────────────┐ │
    │  │ For each session in meta-batch:                         │ │
    │  │                                                         │ │
    │  │   INNER LOOP (Adaptation):                              │ │
    │  │   ┌────────────────────────────────────────────────────┐│ │
    │  │   │ 1. Clone embedding parameters                      ││ │
    │  │   │                                                    ││ │
    │  │   │ 2. For k steps:                                    ││ │
    │  │   │    a. Forward on support set                       ││ │
    │  │   │    b. Compute gradient (NO create_graph)           ││ │
    │  │   │    c. Update embeddings: θ' = θ - α∇L_support      ││ │
    │  │   │                                                    ││ │
    │  │   │ 3. Evaluate on query set with adapted θ'           ││ │
    │  │   └────────────────────────────────────────────────────┘│ │
    │  │                                                         │ │
    │  │   Accumulate query loss for meta-update                 │ │
    │  │   Restore original embeddings                           │ │
    │  └─────────────────────────────────────────────────────────┘ │
    │                                                              │
    │  Update backbone: φ = φ - β∇L_meta                          │
    │  (Uses FIRST-ORDER gradients only)                          │
    └─────────────────────────────────────────────────────────────┘


E2E-TTT Training (Full Second-Order):
=====================================

For each meta-batch:
    ┌─────────────────────────────────────────────────────────────┐
    │  OUTER LOOP (Meta-Update):                                   │
    │  ┌─────────────────────────────────────────────────────────┐ │
    │  │ For each session in meta-batch:                         │ │
    │  │                                                         │ │
    │  │   INNER LOOP (Adaptation with Meta-Gradients):          │ │
    │  │   ┌────────────────────────────────────────────────────┐│ │
    │  │   │ 1. Clone embedding parameters (with gradient)      ││ │
    │  │   │                                                    ││ │
    │  │   │ 2. For k steps:                                    ││ │
    │  │   │    a. Forward on support set                       ││ │
    │  │   │    b. Compute gradient (WITH create_graph=True)    ││ │
    │  │   │    c. Update embeddings: θ' = θ - α∇L_support      ││ │
    │  │   │    (Computation graph preserved!)                  ││ │
    │  │   │                                                    ││ │
    │  │   │ 3. Evaluate on query set with adapted θ'           ││ │
    │  │   └────────────────────────────────────────────────────┘│ │
    │  │                                                         │ │
    │  │   Accumulate query loss (THROUGH inner loop)            │ │
    │  │   Restore original embeddings                           │ │
    │  └─────────────────────────────────────────────────────────┘ │
    │                                                              │
    │  Update ALL params: θ,φ = θ,φ - β∇L_meta                    │
    │  (SECOND-ORDER gradients through adaptation!)               │
    └─────────────────────────────────────────────────────────────┘
```

## Experimental Setup

### Datasets

| Dataset | Species | Sessions | Neurons/Session | Timepoints | Sampling Rate |
|---------|---------|----------|-----------------|------------|---------------|
| `celegansflavell` | C. elegans | 80 | 82-163 | ~1600 | 4 Hz |
| `zebrafishahrens_pc` | Zebrafish | 7 | ~3000 PCs | ~6000 | 1 Hz |

### Data Split

For each session:
- **Training**: First 70% of timepoints
- **Validation**: Next 15% of timepoints
- **Test**: Final 15% of timepoints

For FOMAML/E2E-TTT, each session is further split:
- **Support Set**: 70% of samples (for adaptation)
- **Query Set**: 30% of samples (for meta-update)

### Model Architecture: POCO (Perceiver-IO Conditioning)

```
POCO Model Structure:
====================

Input Layer:
  └── Linear projection: (N_neurons) → (hidden_dim=512)

Embedding Layers:
  ├── Unit Embedding:    (N_total_neurons) → (512)
  ├── Session Embedding: (N_sessions) → (512)
  └── Latent Embedding:  (64 latents) → (512)

Perceiver-IO Core:
  ├── Cross-Attention Encoder: (input) × (latents) → (latents)
  ├── Self-Attention Processor: N_layers × (latents → latents)
  └── Cross-Attention Decoder: (latents) × (queries) → (output)

Output Layer:
  └── Linear projection: (512) → (N_neurons)
```

## Metrics

### Training Metrics

| Metric | Description | Used By |
|--------|-------------|---------|
| `TrainLoss` | MSE on training batch | Baseline |
| `MetaLoss` | Query loss after inner adaptation | FOMAML, E2E-TTT |
| `AvgInnerLoss` | Average loss during inner loop | FOMAML, E2E-TTT |
| `AvgQueryLoss` | Average query loss per session | FOMAML, E2E-TTT |

### Evaluation Metrics (Comparable Across Methods)

| Metric | Description | Comparable? |
|--------|-------------|-------------|
| `TestLoss` | Validation MSE **without** adaptation | Yes |
| `ValLoss` | Validation MSE **with** adaptation (FOMAML/E2E-TTT only) | N/A for baseline |
| `val_mse` | Per-dataset validation MSE | Yes |
| `val_mae` | Per-dataset validation MAE | Yes |
| `val_score` | 1 - (val_mse / chance_mse), higher is better | Yes |

### Key Comparison

```
Fair Comparison:
================

                    POCO Baseline    FOMAML           E2E-TTT
                    -------------    ------           -------
TestLoss            MSE(val)         MSE(val, pre)    MSE(val, pre)
                    (no adaptation)  (no adaptation)  (no adaptation)

ValLoss             N/A              MSE(val, post)   MSE(val, post)
                                     (with adapt)     (with adapt)

Expected Result:
  - TestLoss should be similar across all methods
  - ValLoss (FOMAML/E2E-TTT) should be LOWER than TestLoss
  - Difference = TTT benefit
```

## Loss Functions

### Prediction Loss (All Methods)

```python
loss = MSE(pred[-pred_length:], target[-pred_length:])
```

Where:
- `pred_length = 12` (default) prediction steps
- `seq_length = 24` (default) total sequence length
- Context: `seq_length - pred_length = 12` steps

### Meta-Loss (FOMAML/E2E-TTT)

```python
# After inner loop adaptation on support set
meta_loss = (1/N) * sum(query_loss_i for i in sessions)
```

## Hyperparameters

### Common Settings (All Methods)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_batch` | 5000 | Total training iterations |
| `lr` / `meta_lr` | 1e-4 | Learning rate |
| `hidden_dim` | 512 | Model hidden dimension |
| `num_layers` | 2 | Perceiver processor layers |
| `num_latents` | 64 | Number of latent vectors |

### POCO Baseline

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 64 | Samples per batch (default) |

### FOMAML & E2E-TTT (Aligned)

Both meta-learning methods use identical hyperparameters for fair comparison:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 32 | Samples per batch (reduced for meta-learning) |
| `meta_lr` | 1e-4 | Outer loop learning rate |
| `inner_lr` | 1e-3 | Inner loop learning rate |
| `inner_steps` | 3 | Adaptation steps per session |
| `meta_batch_size` | 2 | Sessions per meta-batch |
| `support_ratio` | 0.7 | Support/query split ratio |

**Key difference:** E2E-TTT uses `use_second_order=True` (create_graph=True) for second-order gradients.

## Objective

The goal of POCO-TTT is to enable **fast adaptation** to new neural recording sessions:

1. **Pre-training**: Learn a good initialization for the backbone using meta-learning
2. **Test-time**: Quickly adapt embeddings to a new session using a few gradient steps

### Success Criteria

- `ValLoss < TestLoss` for FOMAML/E2E-TTT (adaptation helps)
- Comparable `TestLoss` across methods (no degradation from meta-training)
- E2E-TTT should achieve better adaptation than FOMAML (second-order gradients)

## Logging

All experiments log to:
- **Local files**: `experiments/{exp_name}/{model_variant}/progress.txt`
- **WandB**: https://wandb.ai/neuroai/POCO-TTT

### WandB Dashboard

View experiments at: https://wandb.ai/neuroai/POCO-TTT

Key panels to create:
1. **TestLoss vs BatchNum** (all methods) - Training progress comparison
2. **ValLoss vs BatchNum** (FOMAML, E2E-TTT) - Adaptation performance
3. **val_score** - Normalized performance metric

## Running Experiments

```bash
# Submit all experiments to Kempner cluster
sbatch sbatch/run_poco_baseline.sh
sbatch sbatch/run_fomaml.sh
sbatch sbatch/run_e2e_ttt.sh

# Monitor progress
tail -f experiments/*/model_*/progress.txt

# Check WandB
# https://wandb.ai/neuroai/POCO-TTT
```
