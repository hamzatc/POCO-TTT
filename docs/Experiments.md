# POCO-TTT Experiments

## Overview

POCO-TTT implements Test-Time Training (TTT) for neural activity prediction. This document describes the three main training methods being compared and their experimental setup.

## Methods Comparison

### Baselines from POCO Paper

The following baselines from the original POCO paper are included for comparison:

| Model | Description | Type |
|-------|-------------|------|
| **POCO** | Perceiver-IO with population conditioning | Multi-session |
| **NLinear** | Simple linear model per neuron | Multi-session |
| **MLP_L** | MLP with shared latent space | Multi-session |
| **TexFilter** | Texture-based temporal filtering | Multi-session |
| **NetFormer** | Network-aware transformer | Multi-session |
| **Latent_PLRNN** | Piecewise linear RNN with latent space | Multi-session |

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

**Dataset Notes**:
- **C. elegans (Flavell)**: We have 80 sessions vs paper's 40 sessions
- **Zebrafish (Ahrens)**: We have 7 sessions. The public FigShare dataset only contains 7 subjects (1-7). The POCO paper's 15 sessions may be from additional data not publicly available.
- **Data Source**: [Janelia FigShare](https://janelia.figshare.com/articles/dataset/Whole-brain_light-sheet_imaging_data/7272617) (Chen et al., Neuron, 2018)

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
| `{dataset}_val_score` | 1 - (val_mse / copy_mse), higher is better | Yes |
| `PreAdaptScore` | val_score before adaptation (FOMAML/E2E-TTT) | Yes |
| `PostAdaptScore` | val_score after adaptation (FOMAML/E2E-TTT) | Yes |

**Note**: `{dataset}_val_score` is logged per-dataset (e.g., `celegansflavell_val_score`, `zebrafishahrens_pc_val_score`) for baseline training. For FOMAML/E2E-TTT, `PreAdaptScore` and `PostAdaptScore` provide the average val_score across sessions.

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

### WandB Naming Schema

All wandb runs follow a standardized naming convention:

```
{training_mode}_{model}_{dataset}_seed{N}
```

| Component | Description | Examples |
|-----------|-------------|----------|
| `training_mode` | Training mode | `standard`, `fomaml`, `e2e_ttt`, `single_session` |
| `model` | Model architecture | `POCO`, `NLinear`, `MLP`, `TexFilter`, `NetFormer`, etc. |
| `dataset` | Full dataset label | `celegansflavell`, `zebrafishahrens_pc`, `celegansflavell-0` |
| `seed` | Random seed | `seed0`, `seed1`, `seed2` |

**Example Run Names:**
- `standard_POCO_celegansflavell_seed0` - Standard POCO training on C. elegans Flavell
- `fomaml_POCO_celegansflavell_seed1` - FOMAML meta-learning on C. elegans Flavell
- `e2e_ttt_POCO_zebrafishahrens_pc_seed0` - E2E-TTT training on zebrafish Ahrens (PC)
- `single_session_NLinear_celegansflavell-0_seed0` - Single-session NLinear on session 0
- `standard_TexFilter_celegansflavell+zebrafishahrens_pc_seed2` - Multi-dataset training

**Multi-Dataset Naming:**
When training on multiple datasets:
- 2 datasets: `dataset1+dataset2` (e.g., `celegansflavell+zebrafishahrens_pc`)
- 3+ datasets: `dataset1+dataset2+Nmore` (e.g., `celegansflavell+zebrafishahrens_pc+4more`)

**Run Grouping:**
Runs are automatically grouped by `experiment_name` in wandb, making it easy to compare different configurations within the same experiment.

### WandB Dashboard

View experiments at: https://wandb.ai/neuroai/POCO-TTT

Key panels to create:
1. **TestLoss vs BatchNum** (all methods) - Training progress comparison
2. **ValLoss vs BatchNum** (FOMAML, E2E-TTT) - Adaptation performance
3. **{dataset}_val_score** - Normalized performance metric per dataset
4. **PreAdaptScore / PostAdaptScore** - TTT improvement metrics

## Running Experiments

```bash
# Submit all experiments to Kempner cluster
# POCO-TTT comparison
sbatch sbatch/run_poco_baseline.sh      # POCO standard training
sbatch sbatch/run_fomaml.sh             # FOMAML meta-learning
sbatch sbatch/run_e2e_ttt.sh            # E2E-TTT (second-order)

# Paper baselines for comparison
sbatch sbatch/run_paper_baselines.sh    # All baselines (NLinear, MLP_L, TexFilter, NetFormer, Latent_PLRNN)
# Or run individual baselines:
sbatch sbatch/run_nlinear_baseline.sh   # NLinear only (~12h)
sbatch sbatch/run_texfilter_baseline.sh # TexFilter only (~24h)
sbatch sbatch/run_mlp_baseline.sh       # MLP_L only (~24h)

# Monitor progress
tail -f experiments/*/model_*/progress.txt
tail -f sbatch/logs/*.out

# Check WandB
# https://wandb.ai/neuroai/POCO-TTT
```

## Zero-Shot Foundation Model Baseline (Chronos-2)

### Overview

[Chronos-2](https://huggingface.co/amazon/chronos-2) is a pretrained time series foundation model (120M parameters) used as a zero-shot baseline. It requires no training - the model performs inference directly using its pretrained weights.

### Architecture

```
Chronos-2 Pipeline:
==================

Input: Neural activity time series
       (each neuron treated as separate univariate series)
            │
            ▼
    ┌───────────────────┐
    │  Robust Scaling   │
    │  Normalization    │
    └─────────┬─────────┘
              │
              ▼
    ┌───────────────────┐
    │  Patch Embedding  │
    │    (ResNet)       │
    └─────────┬─────────┘
              │
              ▼
    ┌───────────────────┐
    │  Encoder-only     │
    │  Transformer      │
    │  (Group Attention)│
    └─────────┬─────────┘
              │
              ▼
    ┌───────────────────┐
    │  Quantile Output  │
    │  [0.1, 0.5, 0.9]  │
    └─────────┬─────────┘
              │
              ▼
Output: Point prediction (median quantile)
```

### Key Features

- **Zero-shot**: No training required
- **Cross-learning**: Group attention enables information sharing across neurons
- **Probabilistic**: Produces quantile forecasts (we use median for point prediction)
- **Large context**: Up to 8192 tokens context, 1024 prediction steps

### Experiments

| Experiment | Description | Command |
|------------|-------------|---------|
| `chronos2_zeroshot` | Evaluate on all datasets | `python main.py -t chronos2_zeroshot` |
| `chronos2_zeroshot_small` | Use smaller 28M model | `python main.py -t chronos2_zeroshot_small` |
| `compare_chronos2_baseline` | Compare vs trained models | `python main.py -t compare_chronos2_baseline` |

### Running Chronos-2 Experiments

```bash
# Zero-shot evaluation on all datasets
sbatch sbatch/run_chronos2_zeroshot.sh

# Compare Chronos-2 vs trained baselines (NLinear, POCO, MLP)
sbatch sbatch/run_compare_chronos2.sh
```

### Expected Results

Chronos-2 provides a strong zero-shot baseline:
- Should achieve reasonable predictions without any training
- MSE likely between simple baselines (NLinear) and domain-trained models (POCO)
- Useful for assessing the value of domain-specific training vs foundation models

### Model Variants

| Model | Parameters | HuggingFace ID |
|-------|------------|----------------|
| Chronos-2 | 120M | `amazon/chronos-2` |
| Chronos-2-small | 28M | `autogluon/chronos-2-small` |

See `Chronos2.md` in the project root for detailed documentation.

## Single-Session Experiments

Single-session experiments train and evaluate models on individual recording sessions (no cross-session learning). This provides a baseline for how well models can predict neural activity within a single recording.

### Single-Session Models

| Model | Description |
|-------|-------------|
| POCO | Perceiver-IO (single-session variant) |
| NLinear | Simple linear projection per neuron |
| MLP | Multi-layer perceptron |
| TexFilter | Texture-based temporal filtering |
| DLinear | Decomposition Linear model |
| TCN | Temporal Convolutional Network |

### Running Single-Session Experiments

```bash
# C. elegans Flavell (10 sessions x 6 models x 3 seeds = 180 runs)
sbatch sbatch/run_single_session_celegans.sh

# Zebrafish Ahrens (7 sessions x 6 models x 3 seeds = 126 runs)
sbatch sbatch/run_single_session_zebrafish.sh

# All datasets combined
sbatch sbatch/run_single_session_all.sh
```

### Note on Meta-Learning Methods

FOMAML and E2E-TTT are inherently **multi-session** methods - they require multiple sessions/tasks to perform meta-learning. For single-session evaluation:
- Use standard training (POCO baseline) instead
- The benefit of meta-learning appears when adapting to new sessions

## Current Experiment Status

### Multi-Session Experiments

| Experiment | Job ID | Status | Datasets | Models |
|------------|--------|--------|----------|--------|
| `poco_baseline` | 58364939 | Running | celegansflavell, zebrafishahrens_pc | POCO |
| `fomaml_multi_species` | 58364940 | Running | celegansflavell, zebrafishahrens_pc | POCO (meta-learning) |
| `e2e_ttt_multi_species` | 58364941 | Running | celegansflavell, zebrafishahrens_pc | POCO (second-order) |
| `paper_baselines` | 58370238 | Running | celegansflavell, zebrafishahrens_pc | NLinear, MLP_L, TexFilter, NetFormer, Latent_PLRNN |

### Single-Session Experiments

| Experiment | Job ID | Status | Sessions | Models |
|------------|--------|--------|----------|--------|
| `single_session_celegans_flavell` | 58370969 | Running | 10 C. elegans sessions | POCO, NLinear, MLP, TexFilter, DLinear, TCN |
| `single_session_zebrafish_ahrens` | 58370970 | Running | 7 zebrafish sessions | POCO, NLinear, MLP, TexFilter, DLinear, TCN |

## Reference Results (POCO Paper)

From Yu Duan et al., "POCO: Scalable Neural Forecasting through Population Conditioning" (arXiv:2506.14957):

### Multi-Session val_score (MS_POCO)

| Dataset | Paper MS_POCO | Our POCO Baseline |
|---------|---------------|-------------------|
| celegansflavell | 0.213 ± 0.030 | ~0.27 |
| zebrafishahrens_pc | 0.440 ± 0.003 | ~0.46 |

**Note**: Our results are slightly better than the paper, likely due to:
- Different session counts (we have 80 C. elegans sessions vs paper's 40)
- Different data preprocessing or train/val splits

## Dynamical Systems Baselines

### Overview

We compare POCO against classical dynamical systems reconstruction methods that model neural activity as evolving according to latent dynamics. These methods provide interpretable baselines rooted in dynamical systems theory.

### Methods

| Method | Type | Description | Key Parameters |
|--------|------|-------------|----------------|
| **DMD** | Classical | Linear approximation: x_{t+1} = Ax_t via SVD | `svd_rank` (0=optimal) |
| **HODMD** | Classical | Higher-order DMD with delay embedding | `d` (delay), `svd_rank` |
| **EDMD** | Classical | Polynomial lifting for nonlinear dynamics | `poly_degree` (2-3) |
| **KernelEDMD** | Classical | RBF kernel via Random Fourier Features | `n_features`, `gamma` |
| **SINDy** | Classical | Sparse equation discovery with PCA | `threshold`, `n_pca` |
| **KoopmanAE** | Neural | Autoencoder with linear latent dynamics | `latent_dim`, loss weights |

### Theoretical Background

These methods approximate the **Koopman operator**, which linearizes nonlinear dynamics in an infinite-dimensional function space:

```
Koopman Operator Theory:
========================

Nonlinear dynamics:  x_{t+1} = f(x_t)    [nonlinear in state space]
Koopman lifting:     z = φ(x)            [lift to feature space]
Linear dynamics:     z_{t+1} = K z_t     [linear in lifted space]
Reconstruction:      x̂ = φ⁻¹(z)          [project back]

Methods differ in how they approximate φ and K:
- DMD: φ = identity, K via SVD
- EDMD: φ = polynomials, K via least squares
- KernelEDMD: φ approximated via Random Fourier Features
- SINDy: Discovers sparse equations ẋ = Θ(x)ξ
- KoopmanAE: φ, φ⁻¹ learned by neural networks
```

### Running Experiments

```bash
# Classical baselines only (no GPU needed, ~8 hours)
sbatch sbatch/run_classical_baselines.sh

# Koopman Autoencoder test (GPU, ~4 hours)
sbatch sbatch/run_koopman_ae_test.sh

# All dynamical methods including POCO (GPU, ~24 hours)
sbatch sbatch/run_all_dynamical.sh

# Multi-horizon comparison (GPU, ~24 hours)
sbatch sbatch/run_dynamical_horizons.sh
```

### Experiment Configurations

| Experiment | Methods | Datasets | Horizons |
|------------|---------|----------|----------|
| `compare_classical_baselines` | DMD, HODMD, EDMD, KernelEDMD, SINDy | zebrafish_pc, celegansflavell | 16 |
| `koopman_ae_test` | KoopmanAE | zebrafish_pc | 16 |
| `compare_all_dynamical_methods` | All classical + KoopmanAE + POCO + NLinear | zebrafish_pc, celegansflavell | 16 |
| `compare_dynamical_horizons` | DMD, KernelEDMD, KoopmanAE, POCO, NLinear | zebrafish_pc, celegansflavell | 5, 16, 32, 50 |

### Expected Results

Based on dynamical systems theory:

| Method | Short Horizon (5) | Medium (16) | Long (32+) | Notes |
|--------|-------------------|-------------|------------|-------|
| DMD | Good | Moderate | Poor | Best for quasi-linear, short-term |
| HODMD | Good | Good | Moderate | Delay embedding captures more dynamics |
| KernelEDMD | Good | Good | Moderate | Nonlinear lifting helps |
| SINDy | Variable | Variable | Variable | Sensitive to noise, needs PCA |
| KoopmanAE | Good | Good | Good | Neural lifting is flexible |
| POCO | Best | Best | Best | End-to-end learned, no linearity constraint |

### Implementation Details

Classical methods are wrapped as `nn.Module` for compatibility with POCO's evaluation pipeline:

```python
# Classical methods use fit-based training (no gradient descent)
model.fit(train_data)  # SVD, least squares, etc.
pred = model.predict(test_data)  # Multi-step autoregressive

# Neural methods (KoopmanAE) use standard training
optimizer.step()  # Gradient descent on combined loss
```

All methods output predictions in POCO format: `(pred_length, batch_size, n_neurons)`

### Related Documentation

See `docs/DynamicalSystemsMethods.md` for detailed theoretical background and algorithm descriptions.
