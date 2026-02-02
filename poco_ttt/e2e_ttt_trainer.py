"""
E2E-TTT (End-to-End Test-Time Training) Trainer for POCO-TTT

This implements full second-order meta-learning by using create_graph=True
in the inner loop, allowing gradients to flow through the adaptation process.

Key difference from FOMAML:
- FOMAML: Uses first-order gradients only (no create_graph)
- E2E-TTT: Uses second-order gradients (create_graph=True)

This enables learning initialization that is optimized for post-adaptation
performance, not just pre-adaptation loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

from configs.config_global import DEVICE


class E2ETTTTrainer:
    """
    End-to-End Test-Time Training trainer for POCO models.

    Uses full second-order meta-gradients by maintaining computation graphs
    through the inner loop updates. This is more memory intensive than FOMAML
    but can achieve better adaptation performance.

    Training flow:
    1. Sample a batch of sessions
    2. For each session:
       a. Clone parameters with gradient tracking
       b. Inner loop: Adapt with create_graph=True
       c. Evaluate adapted model on query set
    3. Backprop through inner loop to update backbone
    """

    def __init__(
        self,
        model: nn.Module,
        config,
        meta_optimizer: torch.optim.Optimizer,
    ):
        """
        Initialize E2E-TTT trainer.

        Args:
            model: POCO model (e.g., Decoder with POYO)
            config: Configuration with E2E-TTT parameters
            meta_optimizer: Optimizer for all parameters (outer loop)
        """
        self.model = model
        self.config = config
        self.meta_optimizer = meta_optimizer

        self.inner_lr = config.inner_lr
        self.inner_steps = config.inner_steps
        self.accumulation_steps = getattr(config, 'accumulation_steps', 1)

        self.criterion = nn.MSELoss(reduction='mean')

        # Step counter for gradient accumulation
        self.step_count = 0

        # Use gradient checkpointing if configured
        self.use_checkpointing = getattr(config, 'gradient_checkpointing', False)

        # Use mixed precision if configured
        self.use_mixed_precision = getattr(config, 'mixed_precision', False)
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

        # Track training statistics
        self.stats = {
            'meta_losses': [],
            'inner_losses': [],
            'query_losses': [],
        }

    def get_embedding_params(self) -> Dict[str, nn.Parameter]:
        """Get embedding parameters that will be adapted in inner loop"""
        if hasattr(self.model, 'get_ttt_params'):
            ttt_dict = self.model.get_ttt_params()
            params = {}
            for name, module in ttt_dict.items():
                if isinstance(module, nn.Module):
                    for pname, param in module.named_parameters():
                        params[f'{name}.{pname}'] = param
                elif isinstance(module, nn.Parameter):
                    params[name] = module
            return params
        return {}

    def clone_params_with_grad(self) -> Dict[str, torch.Tensor]:
        """Clone embedding parameters with gradient tracking for meta-gradients"""
        emb_params = self.get_embedding_params()
        # Clone with requires_grad=True to track gradients through inner loop
        return {k: v.clone().requires_grad_(True) for k, v in emb_params.items()}

    def set_params(self, params: Dict[str, torch.Tensor]):
        """Set embedding parameters in the model"""
        emb_modules = self.model.get_ttt_params() if hasattr(self.model, 'get_ttt_params') else {}

        for name, value in params.items():
            # Parse the parameter path
            parts = name.split('.')
            if len(parts) == 2:
                module_name, param_name = parts
                if module_name in emb_modules:
                    module = emb_modules[module_name]
                    if hasattr(module, param_name):
                        getattr(module, param_name).data.copy_(value.data)

    def compute_loss(self, batch_data: Tuple, pred_length: int) -> torch.Tensor:
        """
        Compute MSE loss for a batch.

        Args:
            batch_data: (input_list, target_list, info_list)
            pred_length: Number of prediction steps

        Returns:
            Mean loss tensor
        """
        input_list, target_list, info_list = batch_data

        # Forward pass with unit dropout disabled
        output = self.model(input_list, disable_unit_dropout=True)

        total_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        total_count = 0

        for out, tar in zip(output, target_list):
            if tar.numel() == 0:
                continue

            out_pred = out[-pred_length:]
            tar_pred = tar[-pred_length:]

            loss = F.mse_loss(out_pred, tar_pred, reduction='sum')
            total_loss = total_loss + loss
            total_count += tar_pred.numel()

        if total_count > 0:
            return total_loss / total_count
        return total_loss

    def inner_loop_with_meta_gradients(
        self,
        support_data: Tuple,
        pred_length: int,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Run inner loop adaptation with meta-gradient tracking.

        This is the key difference from FOMAML - we use create_graph=True
        to allow backpropagation through the inner loop updates.

        Args:
            support_data: (input_list, target_list, info_list) for support set
            pred_length: Number of prediction steps

        Returns:
            (final_params, inner_losses) - adapted parameters and loss history
        """
        inner_losses = []

        # Get embedding parameters with gradient tracking
        emb_params = self.get_embedding_params()
        if len(emb_params) == 0:
            logging.warning("No embedding parameters found for inner loop")
            return {}, inner_losses

        # Clone parameters for inner loop (with gradient tracking)
        fast_params = {k: v.clone().requires_grad_(True) for k, v in emb_params.items()}

        for step in range(self.inner_steps):
            # Set the fast params in the model temporarily
            self.set_params(fast_params)

            # Compute loss
            loss = self.compute_loss(support_data, pred_length)
            inner_losses.append(loss)

            # Compute gradients with create_graph=True for meta-gradients
            # This is the KEY difference from FOMAML
            grads = torch.autograd.grad(
                loss,
                fast_params.values(),
                create_graph=True,  # Enable second-order gradients
                retain_graph=True,
                allow_unused=True
            )

            # Update fast params (gradient descent step)
            new_fast_params = {}
            for (name, param), grad in zip(fast_params.items(), grads):
                if grad is not None:
                    new_fast_params[name] = param - self.inner_lr * grad
                else:
                    new_fast_params[name] = param

            fast_params = new_fast_params

        # Set final adapted params in model
        self.set_params(fast_params)

        return fast_params, inner_losses

    def meta_train_step(
        self,
        session_data_fn,
        session_ids: List[int],
        pred_length: int,
    ) -> Dict[str, float]:
        """
        Perform one meta-training step with full second-order gradients.

        Args:
            session_data_fn: Function that takes session_id and returns
                            (support_data, query_data) tuples
            session_ids: List of session IDs for this meta-batch
            pred_length: Number of prediction steps

        Returns:
            Dictionary with training statistics
        """
        # Accumulation handling
        if self.step_count % self.accumulation_steps == 0:
            self.meta_optimizer.zero_grad()

        meta_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        all_inner_losses = []
        all_query_losses = []
        valid_sessions = 0

        # Save original state
        original_state = {k: v.clone() for k, v in self.get_embedding_params().items()}

        for session_id in session_ids:
            # Get support and query data for this session
            support_data, query_data = session_data_fn(session_id)

            if support_data is None or query_data is None:
                continue

            # Inner loop with meta-gradient tracking
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    fast_params, inner_losses = self.inner_loop_with_meta_gradients(
                        support_data, pred_length
                    )
            else:
                fast_params, inner_losses = self.inner_loop_with_meta_gradients(
                    support_data, pred_length
                )

            all_inner_losses.extend([l.item() for l in inner_losses])

            # Evaluate on query set with adapted parameters
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    query_loss = self.compute_loss(query_data, pred_length)
            else:
                query_loss = self.compute_loss(query_data, pred_length)

            all_query_losses.append(query_loss.item())

            # E2E-TTT loss: query loss + average inner loss
            # This optimizes for post-adaptation performance
            avg_inner_loss = sum(inner_losses) / len(inner_losses) if inner_losses else torch.tensor(0.0)
            session_meta_loss = query_loss + 0.1 * avg_inner_loss  # Weighted combination

            meta_loss = meta_loss + session_meta_loss
            valid_sessions += 1

            # Restore original embeddings for next session
            self.set_params(original_state)

        if valid_sessions == 0:
            logging.warning("No valid sessions in meta-batch")
            return {
                'meta_loss': 0.0,
                'avg_inner_loss': 0.0,
                'avg_query_loss': 0.0,
            }

        # Average meta loss
        meta_loss = meta_loss / valid_sessions

        # Backward pass with gradient accumulation
        scaled_loss = meta_loss / self.accumulation_steps

        if self.use_mixed_precision:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        self.step_count += 1

        # Step optimizer after accumulation
        if self.step_count % self.accumulation_steps == 0:
            # Gradient clipping
            if hasattr(self.config, 'grad_clip') and self.config.grad_clip is not None:
                if self.use_mixed_precision:
                    self.scaler.unscale_(self.meta_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )

            if self.use_mixed_precision:
                self.scaler.step(self.meta_optimizer)
                self.scaler.update()
            else:
                self.meta_optimizer.step()

        # Track statistics
        stats = {
            'meta_loss': meta_loss.item(),
            'avg_inner_loss': sum(all_inner_losses) / len(all_inner_losses) if all_inner_losses else 0.0,
            'avg_query_loss': sum(all_query_losses) / len(all_query_losses) if all_query_losses else 0.0,
        }

        self.stats['meta_losses'].append(stats['meta_loss'])
        self.stats['inner_losses'].extend(all_inner_losses)
        self.stats['query_losses'].extend(all_query_losses)

        return stats

    def adapt(
        self,
        support_data: Tuple,
        pred_length: int,
        num_steps: Optional[int] = None,
        lr: Optional[float] = None,
    ) -> List[float]:
        """
        Adapt model to a new session at test time.

        At test time, we don't need create_graph=True since we're not
        computing meta-gradients.

        Args:
            support_data: (input_list, target_list, info_list) for adaptation
            pred_length: Number of prediction steps
            num_steps: Number of adaptation steps (default: config.adaptation_steps)
            lr: Learning rate (default: config.adaptation_lr)

        Returns:
            List of adaptation losses
        """
        if num_steps is None:
            num_steps = getattr(self.config, 'adaptation_steps', self.inner_steps)
        if lr is None:
            lr = getattr(self.config, 'adaptation_lr', self.inner_lr)

        # Reset embeddings for new session
        if hasattr(self.model, 'reset_embeddings'):
            self.model.reset_embeddings()

        adaptation_losses = []

        # Get embedding parameters
        emb_params = self.get_embedding_params()
        if len(emb_params) == 0:
            return adaptation_losses

        # Create optimizer for adaptation (no second-order gradients needed)
        optimizer = torch.optim.SGD(list(emb_params.values()), lr=lr)

        for step in range(num_steps):
            optimizer.zero_grad()

            loss = self.compute_loss(support_data, pred_length)
            adaptation_losses.append(loss.item())

            loss.backward()
            optimizer.step()

        return adaptation_losses

    def evaluate_with_adaptation(
        self,
        support_data: Tuple,
        query_data: Tuple,
        pred_length: int,
    ) -> Dict[str, float]:
        """
        Evaluate model with test-time adaptation.

        Args:
            support_data: Data for adaptation
            query_data: Data for evaluation
            pred_length: Number of prediction steps

        Returns:
            Dictionary with evaluation metrics
        """
        # Save original state
        original_state = {k: v.clone() for k, v in self.get_embedding_params().items()}

        # Evaluate without adaptation
        self.model.eval()
        with torch.no_grad():
            pre_adapt_loss = self.compute_loss(query_data, pred_length).item()

        # Adapt on support set
        self.model.train()
        adapt_losses = self.adapt(support_data, pred_length)

        # Evaluate after adaptation
        self.model.eval()
        with torch.no_grad():
            post_adapt_loss = self.compute_loss(query_data, pred_length).item()

        # Restore original state
        self.set_params(original_state)

        return {
            'pre_adapt_loss': pre_adapt_loss,
            'post_adapt_loss': post_adapt_loss,
            'adaptation_losses': adapt_losses,
            'improvement': pre_adapt_loss - post_adapt_loss,
        }

    def get_stats(self) -> Dict[str, List[float]]:
        """Get training statistics"""
        return self.stats

    def reset_stats(self):
        """Reset training statistics"""
        self.stats = {
            'meta_losses': [],
            'inner_losses': [],
            'query_losses': [],
        }
