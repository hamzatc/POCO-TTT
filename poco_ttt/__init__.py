"""
POCO-TTT: Test-Time Training for Neural Prediction

This module implements FOMAML and E2E-TTT training methods for the POCO architecture.
"""

from poco_ttt.fomaml_trainer import FOMAMLTrainer
from poco_ttt.e2e_ttt_trainer import E2ETTTTrainer
from poco_ttt.meta_utils import (
    clone_params,
    compute_loss_with_params,
    get_param_dict,
)

__all__ = [
    'FOMAMLTrainer',
    'E2ETTTTrainer',
    'clone_params',
    'compute_loss_with_params',
    'get_param_dict',
]
