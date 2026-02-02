"""
FOMAML Trainer for POCO-TTT

Implements First-Order MAML for meta-learning on neural prediction tasks.
The key idea is:
- Outer loop: Update backbone (perceiver) to be good for fast adaptation
- Inner loop: Simulate test-time adaptation on embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

from configs.configs import FOMAMLConfig
from configs.config_global import DEVICE


class FOMAMLTrainer:
    """
    First-Order MAML trainer for POCO models.

    Training flow:
    1. Sample a batch of sessions
    2. For each session:
       a. Clone embedding parameters
       b. Inner loop: Adapt embeddings on support set
       c. Evaluate adapted model on query set
    3. Accumulate query losses and update backbone
    """

    def __init__(
        self,
        model: nn.Module,
        config: FOMAMLConfig,
        meta_optimizer: torch.optim.Optimizer,
    ):
        """
        Initialize FOMAML trainer.

        Args:
            model: POCO model (e.g., Decoder with POYO)
            config: FOMAML configuration
            meta_optimizer: Optimizer for backbone parameters (outer loop)
        """
        self.model = model
        self.config = config
        self.meta_optimizer = meta_optimizer

        self.inner_lr = config.inner_lr
        self.inner_steps = config.inner_steps
        self.use_second_order = config.use_second_order

        self.criterion = nn.MSELoss(reduction='mean')

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
            # Extract actual parameters from modules
            params = {}
            for name, module in ttt_dict.items():
                if isinstance(module, nn.Module):
                    for pname, param in module.named_parameters():
                        params[f'{name}.{pname}'] = param
                elif isinstance(module, nn.Parameter):
                    params[name] = module
            return params
        return {}

    def clone_embedding_state(self) -> Dict[str, torch.Tensor]:
        """Clone current embedding parameters"""
        emb_params = self.get_embedding_params()
        return {k: v.detach().clone() for k, v in emb_params.items()}

    def restore_embedding_state(self, state: Dict[str, torch.Tensor]):
        """Restore embedding parameters from saved state"""
        emb_params = self.get_embedding_params()
        for name, param in emb_params.items():
            if name in state:
                param.data.copy_(state[name])

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

        # Forward pass with unit dropout disabled for deterministic inner loop
        output = self.model(input_list, disable_unit_dropout=True)

        total_loss = torch.tensor(0.0, device=DEVICE)
        total_count = 0

        for out, tar in zip(output, target_list):
            if tar.numel() == 0:
                continue

            # Use last pred_length frames for prediction loss
            out_pred = out[-pred_length:]
            tar_pred = tar[-pred_length:]

            loss = F.mse_loss(out_pred, tar_pred, reduction='sum')
            total_loss = total_loss + loss
            total_count += tar_pred.numel()

        if total_count > 0:
            return total_loss / total_count
        return total_loss

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

    def inner_loop(
        self,
        support_data: Tuple,
        pred_length: int,
    ) -> Tuple[Dict[str, torch.Tensor], List[float]]:
        """
        Run inner loop adaptation on support set.

        Uses manual gradient descent (not SGD optimizer) to properly handle
        first-order MAML updates.

        Args:
            support_data: (input_list, target_list, info_list) for support set
            pred_length: Number of prediction steps

        Returns:
            (adapted_params, inner_losses) - adapted parameters and loss history
        """
        inner_losses = []

        # Get embedding parameters
        emb_params = self.get_embedding_params()
        if len(emb_params) == 0:
            logging.warning("No embedding parameters found for inner loop")
            return {}, inner_losses

        # Clone parameters for inner loop
        fast_params = {k: v.clone().requires_grad_(True) for k, v in emb_params.items()}

        # Inner loop steps using manual gradient descent
        for step in range(self.inner_steps):
            # Set the fast params in the model temporarily
            self.set_params(fast_params)

            # Compute loss
            loss = self.compute_loss(support_data, pred_length)
            inner_losses.append(loss.item())

            # Compute gradients manually
            # For FOMAML: create_graph=False (first-order only)
            grads = torch.autograd.grad(
                loss,
                fast_params.values(),
                create_graph=False,  # First-order MAML
                retain_graph=False,
                allow_unused=True
            )

            # Update fast params (gradient descent step)
            new_fast_params = {}
            for (name, param), grad in zip(fast_params.items(), grads):
                if grad is not None:
                    new_fast_params[name] = param - self.inner_lr * grad
                else:
                    new_fast_params[name] = param.clone()

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
        Perform one meta-training step.

        For FOMAML, we:
        1. Adapt embeddings on support set (inner loop)
        2. Compute query loss with adapted embeddings
        3. Use query loss to update ALL model parameters (backbone + embeddings)

        This is first-order because we don't differentiate through the inner loop.

        Args:
            session_data_fn: Function that takes session_id and returns
                            (support_data, query_data) tuples
            session_ids: List of session IDs for this meta-batch
            pred_length: Number of prediction steps

        Returns:
            Dictionary with training statistics
        """
        self.meta_optimizer.zero_grad()

        meta_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        all_inner_losses = []
        all_query_losses = []
        valid_sessions = 0

        # Save original embedding state
        original_state = self.clone_embedding_state()

        for session_id in session_ids:
            # Get support and query data for this session
            support_data, query_data = session_data_fn(session_id)

            if support_data is None or query_data is None:
                continue

            # Inner loop: Adapt embeddings on support set
            # Returns adapted params and loss history
            fast_params, inner_losses = self.inner_loop(support_data, pred_length)
            all_inner_losses.extend(inner_losses)

            # Evaluate on query set with adapted embeddings
            # The model already has adapted params set by inner_loop
            query_loss = self.compute_loss(query_data, pred_length)
            all_query_losses.append(query_loss.item())

            # Accumulate meta loss
            meta_loss = meta_loss + query_loss
            valid_sessions += 1

            # Restore original embeddings for next session
            self.restore_embedding_state(original_state)

        if valid_sessions == 0:
            logging.warning("No valid sessions in meta-batch")
            return {
                'meta_loss': 0.0,
                'avg_inner_loss': 0.0,
                'avg_query_loss': 0.0,
            }

        # Average meta loss
        meta_loss = meta_loss / valid_sessions

        # Backward pass for meta-gradient (updates backbone)
        meta_loss.backward()

        # Gradient clipping
        if hasattr(self.config, 'grad_clip') and self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip
            )

        # Meta optimizer step
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

        This is called during evaluation to adapt the model to a new session's
        data before making predictions.

        Args:
            support_data: (input_list, target_list, info_list) for adaptation
            pred_length: Number of prediction steps
            num_steps: Number of adaptation steps (default: config.adaptation_steps)
            lr: Learning rate (default: config.adaptation_lr)

        Returns:
            List of adaptation losses
        """
        if num_steps is None:
            num_steps = self.config.adaptation_steps
        if lr is None:
            lr = self.config.adaptation_lr

        # Reset embeddings for new session
        if hasattr(self.model, 'reset_embeddings'):
            self.model.reset_embeddings()

        # Store original inner_steps and lr
        orig_inner_steps = self.inner_steps
        orig_inner_lr = self.inner_lr

        # Use adaptation settings
        self.inner_steps = num_steps
        self.inner_lr = lr

        # Run inner loop for adaptation
        _, adaptation_losses = self.inner_loop(support_data, pred_length)

        # Restore original settings
        self.inner_steps = orig_inner_steps
        self.inner_lr = orig_inner_lr

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
        original_state = self.clone_embedding_state()

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
        self.restore_embedding_state(original_state)

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
