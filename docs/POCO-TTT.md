# POCO-TTT: End-to-End Test-Time Training for Neural Forecasting

## Project Overview

This document details the integration of End-to-End Test-Time Training (E2E-TTT) with POCO (POpulation-COnditioned forecaster) for rapid adaptation to new neural recording sessions.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Background](#background)
3. [Compute Resources](#compute-resources)
4. [Current POCO Approach (Baseline)](#current-poco-approach-baseline)
5. [Offline vs Online Adaptation](#offline-vs-online-adaptation)
6. [Implementation Plan](#implementation-plan)
7. [Experiments](#experiments)
8. [Files to Modify/Create](#files-to-modifycreate)
9. [Timeline](#timeline)
10. [Risk Analysis](#risk-analysis)

---

## Executive Summary

**Goal**: Enable POCO to adapt 3-5x faster to new neural recording sessions through meta-learned initialization and test-time training.

**Key Insight**: POCO's forecasting loss (MSE on future neural activity) is structurally identical to language modeling's next-token prediction. E2E-TTT uses the actual task loss for test-time adaptation, making it well-suited for neural forecasting.

**Approach**: Implement a progression of methods:
1. **Baseline**: Standard embedding fine-tuning (already supported)
2. **FOMAML**: First-order MAML for faster prototyping
3. **E2E-TTT**: Full end-to-end test-time training with meta-learning

---

## Background

### What is E2E-TTT?

End-to-End Test-Time Training (E2E-TTT) is a meta-learning approach that:
1. **Meta-trains** initial weights W₀ such that after a few gradient steps at test time, the model performs well
2. **Uses the actual task loss** (not an auxiliary reconstruction loss) for adaptation
3. **Enables continuous adaptation** as new data arrives

```
Training (Outer Loop):
  L(W₀) = E[ Σ_t loss_t(W_t) ]
  where W_t = W_{t-1} - η_inner * ∇loss_t(W_{t-1})

Test Time (Inner Loop):
  For each batch: W = W - η * ∇loss(W)
```

### Why E2E-TTT for POCO?

| Aspect | Language Modeling | Neural Forecasting (POCO) |
|--------|-------------------|---------------------------|
| **Task** | Predict next token | Predict future activity |
| **Loss** | Cross-entropy | MSE |
| **Ground truth** | Unknown at prediction | Known after H timesteps |
| **Adaptation need** | Domain shift | Session variability, neural drift |

Neural forecasting is **better suited** than language modeling because predictions become verifiable ground truth as time passes.

### POCO Architecture Summary

```
INPUT: Neural activity x_{t-H:t} for all neurons
         │
         ▼
┌─────────────────────────────────────┐
│   POYO ENCODER (Perceiver-IO)       │
│                                     │
│   unit_emb + session_emb + input    │
│              │                      │
│              ▼                      │
│   Cross-attention → Self-attention  │
│              │                      │
│              ▼                      │
│       Latent representation         │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│   OUTPUT PROJECTION                 │
│   Query-based decoding              │
└─────────────┬───────────────────────┘
              │
              ▼
OUTPUT: Predicted activity x̂_{t+1:t+H}
```

Key components:
- `unit_emb`: Per-neuron identity embedding
- `session_emb`: Per-session context embedding
- `perceiver_io`: Cross-attention encoder
- Fine-tuning infrastructure: `freeze_backbone`, `embedding_requires_grad()`

---

## Compute Resources

### Harvard Kempner Institute Cluster

| Resource | Specification |
|----------|---------------|
| **A100 GPUs** | 132 x 40GB, HDR Infiniband |
| **H100 GPUs** | 384 x 80GB, 1600 GB/s transfer |
| **Memory per H100 node** | 1.5 TB RAM, 96 cores |

**Recommendation**: Use H100 80GB for E2E-TTT development due to:
- Higher memory for storing computation graphs (meta-gradients)
- Faster training with modern architecture
- Better support for mixed precision

### Memory Considerations for E2E-TTT

E2E-TTT requires `create_graph=True` for meta-gradients, which increases memory:

```python
# Standard training: ~2x model parameters
# E2E-TTT with N inner steps: ~(2 + N) x model parameters

# POCO model size estimate:
# - POYO encoder: ~2M parameters
# - Embeddings: ~100K parameters (varies with sessions)
# - Total: ~10-50 MB model weights

# With 10 inner steps on H100 80GB: Feasible
# With 50 inner steps: May need gradient checkpointing
```

---

## Current POCO Approach (Baseline)

### Current Adaptation: OFFLINE

POCO currently uses **offline adaptation**:

1. **Pre-training**: Train on multi-session data
2. **New session arrives**: Fine-tune embeddings (or full model) for ~200 steps
3. **Deployment**: Freeze weights, run inference

```python
# From configs/configs.py
self.finetuning = False
self.freeze_backbone = False

# From models/poyo/poyo.py
def reset_for_finetuning(self):
    self.unit_emb.reset_parameters()
    self.session_emb.reset_parameters()

def embedding_requires_grad(self, requires_grad=True):
    self.unit_emb.requires_grad_(requires_grad)
    self.session_emb.requires_grad_(requires_grad)
```

### Current Training Loop

```python
# From train.py (simplified)
for epoch in range(config.num_ep):
    for step_ in range(train_data.min_iter_len):
        loss = task_func.roll(net, data, 'train')
        loss.backward()
        optimizer.step()
```

This is standard supervised learning with no meta-learning component.

### Limitations

1. **Manual recalibration**: Requires 200 steps (~15 seconds) for each new session
2. **No continuous adaptation**: Cannot handle neural drift during deployment
3. **No meta-learning**: Initial weights not optimized for fast adaptation

---

## Offline vs Online Adaptation

### Offline Adaptation (Current + Improved)

```
┌─────────────────────────────────────────────────────────────┐
│                    OFFLINE ADAPTATION                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   NEW SESSION ARRIVES                                       │
│          │                                                  │
│          ▼                                                  │
│   ┌─────────────────────────────┐                          │
│   │  Load pre-trained weights   │                          │
│   │  (meta-learned for E2E-TTT) │                          │
│   └──────────────┬──────────────┘                          │
│                  │                                          │
│                  ▼                                          │
│   ┌─────────────────────────────┐                          │
│   │  Adapt on initial data      │                          │
│   │  (10-50 steps vs 200)       │                          │
│   └──────────────┬──────────────┘                          │
│                  │                                          │
│                  ▼                                          │
│   ┌─────────────────────────────┐                          │
│   │  FREEZE weights             │                          │
│   │  Deploy for inference       │                          │
│   └─────────────────────────────┘                          │
│                                                             │
│   PROS:                                                     │
│   - Simple deployment                                       │
│   - Deterministic behavior                                  │
│   - No compute during inference                             │
│                                                             │
│   CONS:                                                     │
│   - Cannot handle neural drift                              │
│   - Requires manual recalibration                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Online Adaptation (New Capability)

```
┌─────────────────────────────────────────────────────────────┐
│                    ONLINE ADAPTATION                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   CONTINUOUS STREAM OF DATA                                 │
│          │                                                  │
│          ▼                                                  │
│   ┌─────────────────────────────────────────────────────┐  │
│   │  Time t: Predict x_{t+1:t+H}                        │  │
│   └──────────────────────┬──────────────────────────────┘  │
│                          │                                  │
│                          ▼                                  │
│   ┌─────────────────────────────────────────────────────┐  │
│   │  Time t+H: Ground truth x_{t+1:t+H} now available   │  │
│   │            Compute loss = MSE(prediction, truth)    │  │
│   └──────────────────────┬──────────────────────────────┘  │
│                          │                                  │
│                          ▼                                  │
│   ┌─────────────────────────────────────────────────────┐  │
│   │  Update weights: W = W - η * ∇loss                  │  │
│   │  Use updated weights for next prediction            │  │
│   └──────────────────────┬──────────────────────────────┘  │
│                          │                                  │
│                          ▼                                  │
│                     (repeat)                                │
│                                                             │
│   PROS:                                                     │
│   - Handles neural drift automatically                      │
│   - No manual recalibration needed                          │
│   - Continuous improvement                                  │
│                                                             │
│   CONS:                                                     │
│   - Compute during inference                                │
│   - H-timestep delay before updates                         │
│   - Potential instability                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Timing Constraint for Online Adaptation

```
Time:    t-H ──────── t ──────── t+H ──────── t+2H
         │           │           │            │
         └──input────┘           │            │
                    └──prediction─┘           │
                                 └──verifiable─┘
```

- At time t: Use x_{t-H:t} to predict x_{t+1:t+H}
- At time t+H: x_{t+1:t+H} is observed → compute true error
- At time t+H: Update weights → improve future predictions

**Critical**: There's an inherent delay of H timesteps before predictions can be verified. For POCO with `pred_length=16`, this is 16 timesteps.

### Recommendation

**Implement both**:
1. Start with **offline adaptation** (simpler, matches current usage)
2. Add **online adaptation** as an extension for real-time applications

---

## Implementation Plan

### Phase 1: Baseline Confirmation

**Goal**: Establish baseline performance for comparison.

```python
# experiments/baseline_finetuning.py

def run_baseline_experiment(config):
    """
    Standard embedding fine-tuning (current POCO approach)
    """
    # Load pre-trained model
    model = load_pretrained_poco(config.pretrained_path)

    # Reset embeddings for new session
    model.reset_for_finetuning()

    # Fine-tune embeddings only
    model.embedding_requires_grad(True)
    for name, param in model.named_parameters():
        if 'emb' not in name:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr
    )

    # Track adaptation curve
    results = {'steps': [], 'loss': [], 'r2': []}

    for step in range(config.max_steps):
        loss = compute_forecasting_loss(model, train_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % config.eval_every == 0:
            eval_loss, r2 = evaluate(model, val_data)
            results['steps'].append(step)
            results['loss'].append(eval_loss)
            results['r2'].append(r2)

    return results
```

**Metrics to measure**:
- Steps to 95% of final performance
- Final R² score
- Adaptation time (wall clock)

---

### Phase 2: FOMAML (First-Order MAML)

**Goal**: Simpler meta-learning baseline without second-order gradients.

FOMAML approximates MAML by ignoring second-order gradient terms, making it more memory-efficient and easier to implement in PyTorch.

```python
# poco_ttt/fomaml_trainer.py

class FOMAMLTrainer:
    """
    First-Order MAML for POCO

    Key insight: FOMAML drops the Hessian term, using only first-order gradients.
    This is much more memory efficient and often works nearly as well.
    """

    def __init__(self, model, config):
        self.model = model
        self.inner_lr = config.inner_lr      # Learning rate for inner loop
        self.meta_lr = config.meta_lr        # Learning rate for outer loop
        self.inner_steps = config.inner_steps  # Steps per inner loop (1-5)

        # Which parameters to update in inner loop
        self.ttt_param_names = ['unit_emb', 'session_emb']

        # Meta-optimizer for outer loop
        self.meta_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.meta_lr
        )

    def get_ttt_params(self):
        """Get parameters that will be updated at test time"""
        return {name: param for name, param in self.model.named_parameters()
                if any(n in name for n in self.ttt_param_names)}

    def inner_loop(self, session_data, params):
        """
        Simulate test-time adaptation on one session
        Returns: final loss after inner loop updates
        """
        # Clone parameters for inner loop
        fast_params = {k: v.clone() for k, v in params.items()}

        for step in range(self.inner_steps):
            # Sample a batch from this session
            batch = sample_batch(session_data)

            # Forward pass with current fast params
            loss = self.compute_loss_with_params(batch, fast_params)

            # Compute gradients w.r.t. fast params
            grads = torch.autograd.grad(loss, fast_params.values())

            # Update fast params (NO create_graph=True, hence first-order)
            fast_params = {
                k: v - self.inner_lr * g
                for (k, v), g in zip(fast_params.items(), grads)
            }

        # Return loss on held-out batch from same session
        query_batch = sample_batch(session_data)
        query_loss = self.compute_loss_with_params(query_batch, fast_params)

        return query_loss

    def compute_loss_with_params(self, batch, params):
        """Compute forecasting loss using given parameters"""
        # Use functional_call for parameter substitution
        predictions = torch.func.functional_call(
            self.model, params, batch['input']
        )
        return F.mse_loss(predictions, batch['target'])

    def meta_train_step(self, session_batch):
        """
        One step of meta-training

        session_batch: List of sessions to meta-train on
        """
        self.meta_optimizer.zero_grad()

        meta_loss = 0.0
        params = self.get_ttt_params()

        for session_data in session_batch:
            # Run inner loop for this session
            query_loss = self.inner_loop(session_data, params)
            meta_loss += query_loss

        meta_loss = meta_loss / len(session_batch)

        # Backward pass for outer loop
        # Note: This only uses first-order gradients because
        # inner_loop doesn't use create_graph=True
        meta_loss.backward()

        self.meta_optimizer.step()

        return meta_loss.item()

    def adapt(self, new_session_data):
        """
        Adapt to new session at test time
        """
        params = self.get_ttt_params()

        for step in range(self.inner_steps):
            batch = sample_batch(new_session_data)
            loss = self.compute_loss_with_params(batch, params)

            grads = torch.autograd.grad(loss, params.values())
            params = {
                k: v - self.inner_lr * g
                for (k, v), g in zip(params.items(), grads)
            }

        # Load adapted parameters back into model
        for name, param in params.items():
            self.model.get_parameter(name).data.copy_(param)
```

**FOMAML vs Full MAML**:
- FOMAML: O(N) memory, where N = model size
- MAML: O(N × K) memory, where K = inner loop steps
- Performance: FOMAML typically achieves 90-95% of MAML performance

---

### Phase 3: Full E2E-TTT

**Goal**: Complete end-to-end test-time training with meta-learning.

```python
# poco_ttt/e2e_ttt_trainer.py

class E2ETTTTrainer:
    """
    Full E2E-TTT for POCO

    Key difference from FOMAML: uses create_graph=True to compute
    gradients through the inner loop updates (second-order gradients).
    """

    def __init__(self, model, config):
        self.model = model
        self.inner_lr = config.inner_lr
        self.meta_lr = config.meta_lr
        self.inner_steps = config.inner_steps

        self.ttt_param_names = config.ttt_params  # Configurable

        self.meta_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.meta_lr,
            weight_decay=config.weight_decay
        )

        # Gradient accumulation for memory efficiency
        self.accumulation_steps = config.accumulation_steps

    def inner_loop_with_meta_gradients(self, session_data, params):
        """
        Inner loop that maintains computation graph for meta-gradients
        """
        # Clone with gradient tracking
        fast_params = {k: v.clone() for k, v in params.items()}

        inner_losses = []

        for step in range(self.inner_steps):
            batch = sample_batch(session_data)

            loss = self.compute_loss_with_params(batch, fast_params)
            inner_losses.append(loss)

            # KEY: create_graph=True to allow backprop through this
            grads = torch.autograd.grad(
                loss,
                fast_params.values(),
                create_graph=True  # This enables meta-gradients
            )

            # Update with gradient tracking maintained
            fast_params = {
                k: v - self.inner_lr * g
                for (k, v), g in zip(fast_params.items(), grads)
            }

        # Query loss for meta-gradient
        query_batch = sample_batch(session_data)
        query_loss = self.compute_loss_with_params(query_batch, fast_params)

        return query_loss, inner_losses

    def meta_train_step(self, session_batch):
        """
        One step of E2E-TTT meta-training
        """
        self.meta_optimizer.zero_grad()

        total_meta_loss = 0.0
        params = dict(self.model.named_parameters())
        ttt_params = {k: v for k, v in params.items()
                      if any(n in k for n in self.ttt_param_names)}

        for i, session_data in enumerate(session_batch):
            query_loss, inner_losses = self.inner_loop_with_meta_gradients(
                session_data, ttt_params
            )

            # Meta loss includes all inner losses (E2E approach)
            # This optimizes W₀ for post-adaptation performance
            meta_loss = query_loss + sum(inner_losses) / len(inner_losses)

            # Accumulate gradients
            (meta_loss / len(session_batch) / self.accumulation_steps).backward()
            total_meta_loss += meta_loss.item()

        # Step optimizer after accumulation
        if (self.step_count + 1) % self.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.meta_optimizer.step()
            self.meta_optimizer.zero_grad()

        self.step_count += 1

        return total_meta_loss / len(session_batch)

    def adapt_online(self, data_stream):
        """
        Online adaptation for continuous deployment

        data_stream: Iterator yielding (past_activity, future_activity) pairs
                    where future_activity becomes available after H timesteps
        """
        params = self.get_ttt_params()

        prediction_buffer = []  # Store predictions until ground truth available

        for t, (x_past, x_future_gt) in enumerate(data_stream):
            # Make prediction with current params
            with torch.no_grad():
                prediction = self.forward_with_params(x_past, params)

            prediction_buffer.append((prediction, x_future_gt))

            # After H timesteps, we can compute loss and update
            if len(prediction_buffer) > self.model.pred_length:
                old_pred, old_gt = prediction_buffer.pop(0)

                # Compute actual forecasting error
                loss = F.mse_loss(old_pred, old_gt)

                # Update TTT parameters
                grads = torch.autograd.grad(loss, params.values())
                params = {
                    k: v - self.inner_lr * g
                    for (k, v), g in zip(params.items(), grads)
                }

            yield prediction
```

### Memory Optimization Techniques

```python
# poco_ttt/memory_utils.py

def gradient_checkpointing_inner_loop(model, session_data, inner_steps, inner_lr):
    """
    Use gradient checkpointing to reduce memory for long inner loops
    """
    from torch.utils.checkpoint import checkpoint

    def inner_step(params, batch):
        loss = compute_loss(model, batch, params)
        grads = torch.autograd.grad(loss, params.values(), create_graph=True)
        new_params = {k: v - inner_lr * g for (k, v), g in zip(params.items(), grads)}
        return new_params, loss

    params = get_ttt_params(model)
    losses = []

    for step in range(inner_steps):
        batch = sample_batch(session_data)
        # Checkpoint to save memory
        params, loss = checkpoint(inner_step, params, batch, use_reentrant=False)
        losses.append(loss)

    return params, losses


def mixed_precision_meta_training(trainer, session_batch):
    """
    Use automatic mixed precision for faster training
    """
    scaler = torch.cuda.amp.GradScaler()

    with torch.cuda.amp.autocast():
        meta_loss = trainer.meta_train_step(session_batch)

    scaler.scale(meta_loss).backward()
    scaler.step(trainer.meta_optimizer)
    scaler.update()
```

---

### Session Chunking Strategy

**Question**: How should sessions be chunked for the inner loop during meta-training?

**Recommendation**: Use **temporal windows within sessions** for the inner loop.

```python
# poco_ttt/data_utils.py

class SessionChunker:
    """
    Chunk sessions into support/query sets for meta-learning

    Each session is divided into temporal windows that simulate
    the test-time adaptation scenario.
    """

    def __init__(self, window_size=1000, overlap=0.5):
        self.window_size = window_size
        self.overlap = overlap

    def chunk_session(self, session_data):
        """
        Chunk a session into windows for meta-learning

        Returns: List of (support_data, query_data) pairs
        """
        total_length = session_data.shape[1]
        stride = int(self.window_size * (1 - self.overlap))

        chunks = []
        for start in range(0, total_length - self.window_size, stride):
            window = session_data[:, start:start + self.window_size]

            # Split window into support (adapt on) and query (evaluate on)
            split_point = int(self.window_size * 0.7)
            support = window[:, :split_point]
            query = window[:, split_point:]

            chunks.append((support, query))

        return chunks

    def create_meta_batch(self, all_sessions, batch_size=4):
        """
        Create a batch of sessions for meta-training

        Each "task" in the meta-batch is adapting to a different session
        """
        selected_sessions = random.sample(all_sessions, batch_size)

        meta_batch = []
        for session in selected_sessions:
            chunks = self.chunk_session(session)
            # Sample one chunk per session
            support, query = random.choice(chunks)
            meta_batch.append({
                'support': support,
                'query': query,
                'session_id': session.id
            })

        return meta_batch
```

**Rationale**:
1. **Temporal windows** preserve the sequential nature of neural data
2. **Support/query split** simulates test-time: adapt on support, evaluate on query
3. **Cross-session batching** trains the model to adapt to diverse sessions

---

## Experiments

### Experiment 1: Baseline Characterization

**Goal**: Establish current POCO adaptation performance.

| Metric | Measurement |
|--------|-------------|
| Steps to 95% performance | ~150-200 |
| Adaptation time | ~15 seconds |
| Final R² | ~0.525 |

### Experiment 2: FOMAML vs Standard Fine-tuning

**Goal**: Validate that meta-learning improves adaptation speed.

```yaml
# configs/fomaml_experiment.yaml
experiment: fomaml_vs_baseline
model:
  type: POYO
  pretrained: checkpoints/poco_pretrained.pth

fomaml:
  inner_lr: 1e-4
  meta_lr: 1e-5
  inner_steps: [1, 3, 5]
  ttt_params: ['unit_emb', 'session_emb']

evaluation:
  metrics: ['steps_to_95', 'final_r2', 'wall_time']
  held_out_sessions: [10, 11, 12, 13, 14]  # 5 sessions for eval
```

**Expected Results**:

| Method | Steps to 95% | Time | Final R² |
|--------|--------------|------|----------|
| Standard fine-tuning | 150-200 | ~15s | 0.525 |
| FOMAML (3 inner steps) | 50-80 | ~5-8s | 0.52-0.53 |

### Experiment 3: Full E2E-TTT

**Goal**: Achieve maximum adaptation speed with full meta-learning.

```yaml
# configs/e2e_ttt_experiment.yaml
experiment: e2e_ttt
model:
  type: POYO
  pretrained: checkpoints/poco_pretrained.pth

e2e_ttt:
  inner_lr: 1e-4
  meta_lr: 1e-5
  inner_steps: 10
  ttt_params: ['unit_emb', 'session_emb']
  accumulation_steps: 4

training:
  meta_batch_size: 4
  total_meta_steps: 10000

memory:
  gradient_checkpointing: true
  mixed_precision: true
```

**Expected Results**:

| Method | Steps to 95% | Time | Final R² |
|--------|--------------|------|----------|
| Standard fine-tuning | 150-200 | ~15s | 0.525 |
| FOMAML | 50-80 | ~5-8s | 0.52 |
| E2E-TTT | **20-40** | **~2-4s** | **0.52-0.55** |

### Experiment 4: Parameter Ablation

**Goal**: Determine optimal parameters to update at test time.

| Config | TTT Parameters | Expected Speed | Expected Ceiling |
|--------|---------------|----------------|------------------|
| A | Embeddings only | Fastest | Moderate |
| B | Embeddings + Latent emb | Medium | Higher |
| C | Embeddings + Perceiver last layer | Slower | Highest |

### Experiment 5: Online vs Offline Adaptation

**Goal**: Compare continuous online adaptation vs one-time offline.

```python
# Simulated neural drift experiment
def simulate_drift_experiment():
    drift_magnitudes = [0.0, 0.1, 0.2, 0.3]  # Gradual drift

    for drift in drift_magnitudes:
        # Apply drift to test data
        test_data = apply_drift(original_test, drift)

        # Offline: Adapt once, freeze
        offline_model = adapt_offline(pretrained, calibration_data)
        offline_results = evaluate(offline_model, test_data)

        # Online: Continuous adaptation
        online_model = copy.deepcopy(pretrained)
        online_results = evaluate_with_online_adaptation(online_model, test_data)

        record(drift, offline_results, online_results)
```

---

## Files to Modify/Create

### New Files to Create

```
poco_ttt/
├── __init__.py
├── config.py              # TTT-specific configuration
├── fomaml_trainer.py      # FOMAML implementation
├── e2e_ttt_trainer.py     # Full E2E-TTT implementation
├── data_utils.py          # Session chunking, meta-batch creation
├── memory_utils.py        # Gradient checkpointing, mixed precision
└── experiments/
    ├── baseline.py        # Baseline fine-tuning experiments
    ├── fomaml_exp.py      # FOMAML experiments
    └── e2e_ttt_exp.py     # E2E-TTT experiments

configs/
└── ttt/
    ├── fomaml.yaml
    └── e2e_ttt.yaml
```

### Existing Files to Modify

| File | Changes |
|------|---------|
| [train.py](train.py) | Add meta-training mode flag |
| [configs/configs.py](configs/configs.py) | Add TTT hyperparameters |
| [models/poyo/poyo.py](models/poyo/poyo.py) | Add `forward_with_params` for functional interface |

### Configuration Additions

```python
# configs/configs.py additions

class TTTConfig:
    def __init__(self):
        # Meta-learning hyperparameters
        self.meta_lr = 1e-5
        self.inner_lr = 1e-4
        self.inner_steps = 5

        # Which parameters to update at test time
        self.ttt_params = ['unit_emb', 'session_emb']

        # Meta-batch settings
        self.meta_batch_size = 4
        self.sessions_per_meta_batch = 4

        # Memory optimization
        self.gradient_checkpointing = False
        self.mixed_precision = True
        self.accumulation_steps = 1

        # Training mode: 'standard', 'fomaml', 'e2e_ttt'
        self.ttt_mode = 'fomaml'
```

---

## Timeline

### Phase 1: Foundation (Weeks 1-2)
- [ ] Implement FOMAML trainer
- [ ] Set up experiment infrastructure
- [ ] Run baseline characterization experiments
- [ ] Validate FOMAML on held-out sessions

### Phase 2: E2E-TTT Implementation (Weeks 3-4)
- [ ] Implement full E2E-TTT trainer
- [ ] Add gradient checkpointing and mixed precision
- [ ] Tune hyperparameters on validation sessions
- [ ] Compare FOMAML vs E2E-TTT

### Phase 3: Experiments (Weeks 5-7)
- [ ] Parameter ablation study
- [ ] Cross-session generalization experiments
- [ ] Online adaptation implementation and evaluation
- [ ] Neural drift simulation experiments

### Phase 4: Documentation & Paper (Weeks 8-10)
- [ ] Compile results
- [ ] Write paper draft
- [ ] Create visualizations
- [ ] Target venue: NeurIPS / ICML / ICLR

---

## Risk Analysis

### Risk 1: Memory Overhead

**Challenge**: Higher-order gradients for E2E-TTT may exceed GPU memory.

**Mitigation**:
1. Start with FOMAML (no second-order gradients)
2. Use gradient checkpointing for longer inner loops
3. Limit inner steps to 5-10 during development
4. Use H100 80GB GPUs

### Risk 2: Meta-Learning Instability

**Challenge**: Meta-learning can be unstable, especially with long inner loops.

**Mitigation**:
1. Start with short inner loops (3-5 steps)
2. Use gradient clipping
3. Warm up learning rates
4. Monitor meta-gradient norms

### Risk 3: Overfitting to Training Sessions

**Challenge**: Meta-learning might overfit to specific session characteristics.

**Mitigation**:
1. Hold out sessions for validation
2. Use diverse sessions in meta-batches
3. Regularize embeddings
4. Early stopping based on held-out performance

### Risk 4: Online Adaptation Delay

**Challenge**: H-timestep delay before predictions can be verified.

**Mitigation**:
1. Use prediction buffer to manage delay
2. Consider hybrid: rapid initial adaptation + slower continuous updates
3. For real-time applications, prioritize offline adaptation

---

## References

1. E2E-TTT Paper: [End-to-End Test-Time Training](https://github.com/test-time-training/e2e)
2. TTT-LM-PyTorch: [TTT Linear PyTorch Implementation](https://github.com/test-time-training/ttt-lm-pytorch)
3. MAML: [Model-Agnostic Meta-Learning](https://arxiv.org/abs/1703.03400)
4. POCO Paper: Original POCO publication
5. Harvard Kempner Cluster: [Kempner HPC Handbook](https://handbook.eng.kempnerinstitute.harvard.edu/)

---

*Document Version: 1.0*
*Created: January 2026*
*Last Updated: January 2026*
