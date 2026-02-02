# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

POCO-TTT is a neural prediction framework for modeling neural activity across multiple brain regions and organisms (zebrafish, C. elegans, mice). It implements various temporal models (RNNs, TCNs, Transformers) for autoregressive neural activity prediction.

## Commands

```bash
# Environment setup
conda env create -f environment.yml
conda activate poco

# Training
python main.py -t exp_name                     # Train single experiment
python main.py -t exp1 exp2 exp3               # Train multiple experiments
python main.py -t exp_name -s                  # Multi-GPU server mode
python main.py -t exp_name -c -p partition     # Slurm cluster submission

# Evaluation and analysis
python main.py -e exp_name                     # Evaluate trained models
python main.py -a exp_name                     # Run analysis functions

# Data preprocessing (required before training)
python run_preprocess.py
```

## Architecture

### Configuration System
- `configs/configs.py`: Hierarchical config classes (`BaseConfig` → `SupervisedLearningBaseConfig` → `NeuralPredictionConfig`)
- `configs/experiments.py`: Experiment definitions as functions returning config dictionaries
- `configs/configure_model_datasets.py`: Auto-configures model/dataset parameters based on selections
- `configs/config_global.py`: Global paths, seeds (NP_SEED=233, TCH_SEED=3407), device settings

### Training Pipeline
`main.py` (CLI) → `train.py` (model_train) → DatasetIters → TaskFunction → training loop

### Model Types
- **Single-session** (`models/single_session_models.py`): `Autoregressive`, `TCN`, `DLinear`, `TSMixer`, `NLinear`, `MLP`
- **Multi-session** (`models/multi_session_models.py`): Shared latent space wrappers with per-session projections
- **POYO** (`models/poyo/`): Attention-based temporal model with rotary embeddings
- **RNN types** (`models/layers/rnns.py`): LSTM, GRU, CTRNN, PLRNN, LRRNN

### Dataset Architecture
- Base: `NeuralDataset` in `datasets/datasets.py` with `load_all_activities()` method
- Implementations: `Zebrafish`, `ZebrafishAhrens`, `Celegans`, `CelegansFlavell`, `Mice`, `Simulation`
- Data format: Neural activity matrices (N neurons × T timesteps)

### Task/Loss System
- `tasks/taskfunctions.py`: `TaskFunction` base class, `NeuralPrediction` task
- MSE loss by default, supports teacher forcing and multi-step predictions

## Directory Structure

- `experiments/{exp_name}/{model_name}/`: Training outputs, saved models, predictions
- `data/raw_*` and `data/processed_*`: Raw and preprocessed neural data
- `figures/`: Analysis outputs
- `sbatch/`: Slurm job logs

## Adding New Experiments

1. Define experiment function in `configs/experiments.py` returning config dict
2. Use `vary_config()` from `utils/config_utils.py` for parameter sweeps
3. Run with `python main.py -t your_exp_name`

## Adding Custom Datasets

1. Inherit from `NeuralDataset` in `datasets/datasets.py`
2. Implement `load_all_activities()` method
3. Add dataset initialization in `datasets/dataloader.py`
4. Add configuration options in `configs/configure_model_datasets.py`
