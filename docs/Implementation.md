# POCO-TTT Implementation Summary

## Dataset Status

**Current Status: All datasets downloaded and preprocessed**

Data directory: `/n/netscratch/pehlevan_lab/Lab/hamza/neural` (symlinked from `./data`)

| Dataset | Source | Sessions | Status |
|---------|--------|----------|--------|
| Zebrafish (Ahrens) | [Zenodo](https://zenodo.org/records/15130668) | 7 | ✅ Downloaded & Preprocessed |
| C. elegans (Zimmer) | [GitHub](https://github.com/akshey-kumar/BunDLe-Net/tree/main/data/raw) | 5 | ✅ Downloaded & Preprocessed |
| C. elegans (Flavell) | [WormWideWeb](https://wormwideweb.org/activity/dataset/) | 80 | ✅ Downloaded & Preprocessed |
| Mice | IBL dataset | 12 | Requires separate access |

---

## Overview

POCO-TTT extends the POCO (Population-level neural dynamics model with COntextual encoding) framework with Test-Time Training (TTT) capabilities using:

1. **FOMAML (First-Order MAML)**: First-order meta-learning for efficient adaptation
2. **E2E-TTT (End-to-End Test-Time Training)**: Full second-order meta-learning with create_graph=True

This enables faster adaptation to new neural recording sessions by meta-learning a backbone that can quickly adapt via gradient-based fine-tuning of session-specific embeddings.

## Architecture

### Training Paradigms

**Standard POCO Training:**
```
Input → Embeddings → Perceiver → Output Projection → Prediction
         (frozen after pretraining)
```

**POCO-TTT (FOMAML) Training:**
```
Outer Loop (Meta-training):
  For each meta-batch of sessions:
    Inner Loop (Adaptation simulation):
      - Clone embedding parameters
      - Adapt on support set (K gradient steps)
      - Evaluate on query set
    Accumulate query losses → Update backbone

Test Time:
  - Reset embeddings for new session
  - Adapt embeddings on available data
  - Make predictions with adapted model
```

**E2E-TTT Training (Full Second-Order):**
```
Same as FOMAML but with create_graph=True in inner loop
→ Enables gradients through the adaptation process
→ More memory intensive but potentially better initialization
```

### Key Insight

The hybrid approach separates parameters into:
- **Backbone (meta-learned)**: Perceiver layers, output projections - learned to enable fast adaptation
- **Embeddings (adapted at test time)**: Unit embeddings, session embeddings - quickly adapted to new sessions

## Implementation Details

### Files Modified

| File | Changes |
|------|---------|
| `models/poyo/poyo.py` | Added `disable_unit_dropout` flag, `get_ttt_params()`, `get_backbone_params()` |
| `models/multi_session_models.py` | Added TTT helper methods to Decoder class |
| `configs/configs.py` | Added `FOMAMLConfig` and `E2ETTTConfig` classes |
| `datasets/dataloader.py` | Added `SessionDatasetIters` class |
| `train.py` | Added `fomaml_train()`, `e2e_ttt_train()`, evaluation functions |
| `configs/experiments.py` | Added FOMAML and E2E-TTT experiment configurations |
| `utils/logger.py` | Added wandb integration |

### New Files Created

| File | Purpose |
|------|---------|
| `poco_ttt/__init__.py` | Module initialization |
| `poco_ttt/fomaml_trainer.py` | Core FOMAML training logic |
| `poco_ttt/e2e_ttt_trainer.py` | E2E-TTT training with second-order gradients |
| `poco_ttt/meta_utils.py` | Utility functions for meta-learning |
| `pyproject.toml` | uv package configuration |
| `sbatch/run_*.sh` | Slurm submission scripts for Kempner cluster |

## Configuration

### FOMAMLConfig Parameters

```python
class FOMAMLConfig(NeuralPredictionConfig):
    # Training mode
    training_mode = 'fomaml'      # 'standard', 'fomaml', 'e2e_ttt'

    # Meta-learning hyperparameters
    meta_lr = 1e-4                # Outer loop learning rate (backbone)
    inner_lr = 1e-3               # Inner loop learning rate (embeddings)
    inner_steps = 5               # Gradient steps in inner loop
    meta_batch_size = 4           # Sessions per meta-batch

    # Data splitting
    support_ratio = 0.7           # Fraction for support set

    # Second-order gradients (E2E-TTT)
    use_second_order = False      # If True, uses create_graph=True

    # Test-time adaptation
    adaptation_steps = 10         # Steps for test-time adaptation
    adaptation_lr = 1e-3          # Learning rate for adaptation
```

### E2ETTTConfig Parameters

```python
class E2ETTTConfig(FOMAMLConfig):
    training_mode = 'e2e_ttt'
    use_second_order = True       # Enable second-order gradients

    # Memory optimization
    gradient_checkpointing = False
    mixed_precision = False
    accumulation_steps = 1

    # Typically fewer steps due to memory
    inner_steps = 3
    meta_batch_size = 2
```

## Usage

### Environment Setup

```bash
# Create and activate virtual environment
uv venv --python 3.10
source .venv/bin/activate
uv sync
```

### Running Experiments Locally

```bash
# Test FOMAML on small dataset
python main.py -t fomaml_test

# Test E2E-TTT on small dataset
python main.py -t e2e_ttt_test

# POCO baseline for comparison
python main.py -t poco_baseline

# Compare methods on multiple datasets
python main.py -t compare_ttt_methods
```

### Running on Kempner Cluster

```bash
# Submit all experiments
./sbatch/run_all_experiments.sh

# Or submit individually
sbatch sbatch/run_poco_baseline.sh
sbatch sbatch/run_fomaml.sh
sbatch sbatch/run_e2e_ttt.sh
sbatch sbatch/run_compare_ttt_methods.sh

# Monitor jobs
squeue -u $USER

# View logs
tail -f sbatch/logs/*.out
```

## Wandb Integration

All experiments log to Weights & Biases at: https://wandb.ai/neuroai/POCO-TTT

Environment variables (automatically set in sbatch scripts):
```bash
export WANDB_PROJECT=POCO-TTT
export WANDB_ENTITY=neuroai
```

Metrics logged:
- `MetaLoss`: Combined meta-learning loss
- `AvgInnerLoss`: Average loss during inner loop
- `AvgQueryLoss`: Loss on query set after adaptation
- `ValLoss`: Validation loss with adaptation
- `TrainLoss`: Standard training loss (for baseline)
- `TestLoss`: Test loss

## Algorithm

### FOMAML Training Step

```python
def meta_train_step(session_ids):
    meta_loss = 0

    for session_id in session_ids:
        # Get support/query split
        support_data, query_data = get_split(session_id)

        # Save original embeddings
        original_state = clone_embedding_state()

        # Inner loop: adapt embeddings on support
        for step in range(inner_steps):
            loss = compute_loss(support_data)
            # FOMAML: create_graph=False (first-order only)
            grads = torch.autograd.grad(loss, embedding_params)
            embedding_params -= inner_lr * grads

        # Evaluate on query with adapted embeddings
        query_loss = compute_loss(query_data)
        meta_loss += query_loss

        # Restore embeddings for next session
        restore_embedding_state(original_state)

    # Update backbone with meta-gradient
    meta_loss.backward()
    meta_optimizer.step()
```

### E2E-TTT Training Step

Same as FOMAML but with:
```python
# E2E-TTT: create_graph=True (second-order gradients)
grads = torch.autograd.grad(loss, embedding_params, create_graph=True)
```

This enables backpropagation through the inner loop updates, optimizing for post-adaptation performance.

### Test-Time Adaptation

```python
def adapt(support_data, num_steps=10):
    # Reset embeddings for new session
    model.reset_embeddings()

    # Adapt on available data
    for step in range(num_steps):
        loss = compute_loss(support_data)
        loss.backward()
        embedding_optimizer.step()
```

## Key Classes

### FOMAMLTrainer (`poco_ttt/fomaml_trainer.py`)

Main class for FOMAML training:

- `__init__(model, config, meta_optimizer)`: Initialize trainer
- `inner_loop(support_data, pred_length)`: Run inner loop adaptation
- `meta_train_step(session_data_fn, session_ids, pred_length)`: One meta-training step
- `adapt(support_data, pred_length, num_steps, lr)`: Test-time adaptation
- `evaluate_with_adaptation(support_data, query_data, pred_length)`: Evaluate with TTT

### E2ETTTTrainer (`poco_ttt/e2e_ttt_trainer.py`)

Main class for E2E-TTT training (second-order):

- Same interface as FOMAMLTrainer
- Uses `create_graph=True` in inner loop
- Supports gradient checkpointing and mixed precision

### SessionDatasetIters (`datasets/dataloader.py`)

Session-level data access for meta-learning:

- `get_session(session_id)`: Get all data for a session
- `get_support_query_split(session_id, support_ratio)`: Split into support/query
- `sample_meta_batch(batch_size)`: Sample sessions for meta-batch
- `create_batch_from_data(data_list, session_id)`: Create model input batch

## Available Experiments

| Experiment | Description |
|------------|-------------|
| `poco_baseline` | Standard POCO training (no meta-learning) |
| `fomaml_test` | Quick FOMAML test on C. elegans |
| `fomaml_zebrafish` | FOMAML on zebrafish dataset |
| `fomaml_multi_species` | FOMAML across multiple species |
| `fomaml_compare_inner_steps` | Compare different inner loop steps |
| `fomaml_compare_inner_lr` | Compare different inner learning rates |
| `e2e_ttt_test` | Quick E2E-TTT test on C. elegans |
| `e2e_ttt_zebrafish` | E2E-TTT on zebrafish dataset |
| `e2e_ttt_multi_species` | E2E-TTT across multiple species |
| `e2e_ttt_compare_inner_steps` | Compare inner steps for E2E-TTT |
| `compare_ttt_methods` | Compare FOMAML vs E2E-TTT |

## Expected Results

Based on the feasibility analysis, POCO-TTT should provide:

1. **Faster adaptation**: 2-5x fewer gradient steps to reach equivalent performance
2. **Better few-shot performance**: Improved prediction with limited adaptation data
3. **Cross-session generalization**: Better transfer to unseen recording sessions

## Memory Considerations

| Mode | Memory Usage | Notes |
|------|--------------|-------|
| Standard training | ~2× params | Weights + gradients |
| FOMAML | ~2× params + batch memory | No computation graph storage |
| E2E-TTT | ~(2 + inner_steps) × params | Stores graphs for each inner step |

For H100 80GB GPUs, all modes are feasible with standard configurations.

## Evaluation Metrics

The key metrics to compare:
1. **Final MSE**: Prediction error after full training
2. **Adaptation Speed**: Steps to reach target MSE on new session
3. **Few-shot Performance**: MSE with limited adaptation data (e.g., 10, 50, 100 samples)
4. **Pre/Post Adaptation Loss**: Improvement from test-time training

Results will be saved in `experiments/{exp_name}/` with:
- Training logs in `progress.txt`
- Best model in `net_best.pth`
- Validation predictions in `val_pred_vs_target/`

## References

- MAML: Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" (2017)
- FOMAML: First-order approximation of MAML
- E2E-TTT: Sun et al., "Test-Time Training" papers
- POCO: Original POCO paper for neural population dynamics modeling
