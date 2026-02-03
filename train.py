import logging
import os.path as osp
import random
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lrs
import time
from typing import Dict

from configs.config_global import LOG_LEVEL, NP_SEED, TCH_SEED, USE_CUDA, DATA_DIR
from configs.configs import BaseConfig, SupervisedLearningBaseConfig, NeuralPredictionConfig, FOMAMLConfig
from datasets.dataloader import DatasetIters, SessionDatasetIters
from tasks.taskfunctions import TaskFunction
from models.model_utils import model_init, is_classical_baseline
from utils.config_utils import load_config
from utils.logger import Logger
from utils.train_utils import (get_grad_norm, grad_clipping, task_init, log_complete)
from datetime import datetime

def train_from_path(path):
    """Train from a path with a config file in it."""
    logging.basicConfig(level=LOG_LEVEL)
    config = load_config(path)
    model_train(config)

def eval_from_path(path):
    """Evaluate from a path with a config file in it."""
    logging.basicConfig(level=LOG_LEVEL)
    config = load_config(path)
    model_eval(config)

def evaluate_performance(
    net: nn.Module, 
    config: SupervisedLearningBaseConfig, 
    test_data: DatasetIters, 
    task_func: TaskFunction, 
    logger: Logger, 
    testloss_list: list = [], 
    i_b: int = 0,
    train_loss: float = 0,
    phase: str = 'val',
    testing: bool = False
):
    """
    Test the model. Print the test loss and accuracy through logger. Save the model if it is the best model.
    """

    net.eval()
    start = time.time()
    with torch.no_grad():

        test_data.reset()
        all_loss = 0

        for i_tloader, test_iter in enumerate(test_data.data_iters):
            total = 0
            test_loss = 0.0
            while True:
                try:
                    t_data = next(test_iter)
                    t_data = get_full_input(t_data, i_tloader, config, test_data.input_sizes)
                    result = task_func.roll(net, t_data, phase)
                    loss, num = result[: 2]
                    test_loss += loss.item() * num
                    total += num
                except StopIteration:
                    break
            dataset_loss = test_loss / total
            if config.log_loss:
                dataset_loss = np.log(dataset_loss + 1e-4)
            all_loss += dataset_loss * config.mod_w[i_tloader]
        avg_testloss = all_loss / sum(config.mod_w)
    
    eval_time = time.time() - start
    logging.info('Avg inference time per batch: {:.1f} ms'.format(eval_time * 1000 / len(test_data.data_loaders[0].dataset) * config.batch_size))
    logger.log_tabular('val/loss', avg_testloss)
    testloss_list.append(avg_testloss)

    # save the model with best testing loss
    if not testing:
        best = False
        if avg_testloss <= min(testloss_list):
            best = True
            torch.save(net.state_dict(), osp.join(config.save_path, 'net_best.pth'))
    else:
        best = True

    task_func.after_testing_callback(
        logger=logger, save_path=config.save_path, is_best=best, batch_num=i_b, testing=testing
    )

def get_full_input(batch, dataset_idx, config: NeuralPredictionConfig, input_sizes):
    """
    Use the input list for one dataset to create the full input list for all datasets.
    """
    # prepare an empty list
    input_size_list = list(itertools.chain(*input_sizes))
    n_sessions = [len(input_size) for input_size in input_sizes]
    dataset_start_idx = [sum(n_sessions[:i]) for i in range(len(n_sessions) + 1)]

    batch_input, batch_target, batch_info = batch

    input_list = []
    target_list = []
    info_list = []

    for i, size in enumerate(input_size_list):
        if i >= dataset_start_idx[dataset_idx] and i < dataset_start_idx[dataset_idx + 1]:
            idx = i - dataset_start_idx[dataset_idx]
            input_list.append(batch_input[idx])
            target_list.append(batch_target[idx])
            info_list.append(batch_info[idx])
        else:
            input_list.append(torch.zeros(config.seq_length - config.pred_length, 0, size))
            target_list.append(torch.zeros(config.seq_length - 1, 0, size))
            info_list.append({'session_idx': i, })

    return input_list, target_list, info_list

def model_train(config: NeuralPredictionConfig):
    """
    The main training function.
    This function initializes the task, dataset, network, optimizer, and learning rate scheduler.
    It then trains the network and logs the performance.
    """

    # Route to inference-only evaluation for zero-shot models (e.g., Chronos-2)
    if config.model_type == 'Chronos2':
        return inference_only_eval(config)

    # Route to classical dynamics baseline training (DMD, SINDy, etc.)
    if is_classical_baseline(config):
        return classical_baseline_train(config)

    # Route to FOMAML training if configured
    if hasattr(config, 'training_mode') and config.training_mode == 'fomaml':
        return fomaml_train(config)

    # Route to E2E-TTT training if configured
    if hasattr(config, 'training_mode') and config.training_mode == 'e2e_ttt':
        return e2e_ttt_train(config)

    total_train_time = 0.0
    start_time = time.time()

    np.random.seed(NP_SEED + config.seed)
    torch.manual_seed(TCH_SEED + config.seed)
    random.seed(config.seed)
    # set the torch hub directory
    torch.hub.set_dir(osp.join(DATA_DIR, 'torch_hub'))
    train_start_time = datetime.now()

    if USE_CUDA:
        logging.info("training with GPU")

    # initialize logger with wandb support
    logger = Logger(output_dir=config.save_path,
                    exp_name=config.experiment_name,
                    config=config)

    # initialize dataset
    train_data = DatasetIters(config, 'train')
    assert config.perform_val
    test_data = DatasetIters(config, 'val')
    test_baseline_performance = test_data.get_baselines()

    # initialize task
    task_func: TaskFunction = task_init(config, train_data.input_sizes)
    task_func.mse_baseline['val'] = test_baseline_performance

    # initialize model
    net = model_init(config, train_data.input_sizes, train_data.unit_types)
    param_count = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Model: {}, Total parameters: {}'.format(config.model_type, param_count))
    
    # initialize optimizer
    if config.optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.wdecay)
    elif config.optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.wdecay, amsgrad=True)
    elif config.optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=config.lr_SGD,
                                    momentum=0.9, weight_decay=config.wdecay)
    else:
        raise NotImplementedError('optimizer not implemented')

    # initialize Learning rate scheduler
    if config.use_lr_scheduler:
        if config.scheduler_type == 'ExponentialLR':
            scheduler = lrs.ExponentialLR(optimizer, gamma=0.1)
        elif config.scheduler_type == 'StepLR':
            scheduler = lrs.StepLR(optimizer, 1.0, gamma=0.1)
        elif config.scheduler_type == 'CosineAnnealing':
            scheduler = lrs.CosineAnnealingLR(
                optimizer, 
                T_max=config.max_batch // config.log_every + 1,
                eta_min=config.lr / 20
            )
        else:
            raise NotImplementedError('scheduler_type must be specified')

    i_b = 0
    i_log = 0
    testloss_list = []
    break_flag = False
    train_loss = 0.0

    for epoch in range(config.num_ep):
        train_data.reset()

        for step_ in range(train_data.min_iter_len):
            
            net.train()
            # save model
            if (i_b + 1) % config.save_every == 0:
                torch.save(net.state_dict(), osp.join(config.save_path, 'net_{}.pth'.format(i_b + 1)))

            loss = 0.0
            optimizer.zero_grad()

            for i_loader, train_iter in enumerate(train_data.data_iters):
                mod_weight = config.mod_w[i_loader]
                data = next(train_iter)
                data = get_full_input(data, i_loader, config, train_data.input_sizes)
                dataset_loss = task_func.roll(net, data, 'train')
                if config.log_loss:
                    dataset_loss = torch.log(dataset_loss + 1e-4)
                loss += dataset_loss * mod_weight

            loss.backward()
            # gradient clipping
            if config.grad_clip is not None:
                grad_clipping(net, config.grad_clip)

            optimizer.step()
            train_loss += loss.item()            

            # log performance
            if i_b % config.log_every == config.log_every - 1:

                logger.log_tabular('train/epoch', epoch)
                logger.log_tabular('train/batch_number', i_b + 1)
                logger.log_tabular('train/samples_seen', (i_b + 1) * config.batch_size * train_data.num_datasets)
                logger.log_tabular('train/loss', train_loss / config.log_every)

                evaluate_performance(net, config, test_data, task_func, logger, testloss_list, i_b + 1, train_loss / config.log_every)
                i_log += 1
                logger.dump_tabular()
                train_loss = 0

                if config.use_lr_scheduler:
                    scheduler.step()

            i_b += 1
            if i_b >= config.max_batch:
                break_flag = True
                break

        if break_flag:
            break

    total_train_time += time.time() - start_time
    logging.info('Total training time: {:.2f} mins'.format(total_train_time / 60))

    task_func.after_training_callback(config, net)
    log_complete(config.save_path, train_start_time)

    if config.perform_test:
        model_eval(config)

def model_eval(config: SupervisedLearningBaseConfig):
    np.random.seed(NP_SEED)
    torch.manual_seed(TCH_SEED)
    random.seed(config.seed)

    test_data = DatasetIters(config, 'test')
    task_func: TaskFunction = task_init(config, test_data.input_sizes)
    task_func.mse_baseline['val'] = test_data.get_baselines()
    net = model_init(config, test_data.input_sizes, test_data.unit_types)
    net.load_state_dict(torch.load(osp.join(config.save_path, 'net_best.pth'), weights_only=True))

    logger = Logger(output_dir=config.save_path, output_fname='test.txt', exp_name=config.experiment_name)
    evaluate_performance(net, config, test_data, task_func, logger, testing=True)
    logger.dump_tabular()


def inference_only_eval(config: NeuralPredictionConfig):
    """
    Evaluation-only pipeline for pretrained zero-shot models like Chronos-2.
    No training is performed - the model is evaluated directly on the validation set.
    """
    np.random.seed(NP_SEED + config.seed)
    torch.manual_seed(TCH_SEED + config.seed)
    random.seed(config.seed)
    torch.hub.set_dir(osp.join(DATA_DIR, 'torch_hub'))

    if USE_CUDA:
        logging.info("Zero-shot evaluation with GPU")

    # Initialize logger
    logger = Logger(output_dir=config.save_path, exp_name=config.experiment_name, config=config)

    # Initialize datasets (train_data needed for proper input_sizes/unit_types)
    train_data = DatasetIters(config, 'train')
    test_data = DatasetIters(config, 'val')
    test_baseline_performance = test_data.get_baselines()

    # Initialize task
    task_func: TaskFunction = task_init(config, train_data.input_sizes)
    task_func.mse_baseline['val'] = test_baseline_performance

    # Initialize model (no training will occur)
    net = model_init(config, train_data.input_sizes, train_data.unit_types)
    param_count = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logging.info(f'Model: {config.model_type}, Total parameters: {param_count}')
    logging.info(f'Zero-shot model - skipping training, evaluating directly')

    # Evaluate performance
    testloss_list = [float('inf')]
    evaluate_performance(
        net, config, test_data, task_func, logger,
        testloss_list=testloss_list, i_b=0, train_loss=0.0
    )
    logger.dump_tabular()
    logging.info("Zero-shot evaluation complete.")


def classical_baseline_train(config: NeuralPredictionConfig):
    """
    Training pipeline for classical dynamics baselines (DMD, SINDy, etc.).

    These methods use closed-form solutions rather than gradient descent:
    1. Collect training data
    2. Fit model using analytical solution (SVD, sparse regression, etc.)
    3. Evaluate on validation set
    """
    total_train_time = 0.0
    start_time = time.time()

    np.random.seed(NP_SEED + config.seed)
    torch.manual_seed(TCH_SEED + config.seed)
    random.seed(config.seed)
    torch.hub.set_dir(osp.join(DATA_DIR, 'torch_hub'))
    train_start_time = datetime.now()

    if USE_CUDA:
        logging.info(f"Classical baseline training: {config.model_type}")

    # Initialize logger
    logger = Logger(output_dir=config.save_path, exp_name=config.experiment_name, config=config)

    # Initialize datasets
    train_data = DatasetIters(config, 'train')
    test_data = DatasetIters(config, 'val')
    test_baseline_performance = test_data.get_baselines()

    # Initialize model
    net = model_init(config, train_data.input_sizes, train_data.unit_types)
    logging.info(f'Model: {config.model_type}')

    # Initialize task function for evaluation
    task_func: TaskFunction = task_init(config, train_data.input_sizes)
    task_func.mse_baseline['val'] = test_baseline_performance

    # Collect training data and fit model for each session
    logging.info(f"Fitting {config.model_type} on training data...")
    fit_start_time = time.time()

    input_size_list = list(itertools.chain(*train_data.input_sizes))
    n_sessions = len(input_size_list)

    for session_idx in range(n_sessions):
        session_chunks = []

        # Collect data for this session from all batches
        for loader in train_data.data_loaders:
            for batch in loader:
                batch_input, batch_target, batch_info = batch
                batch = get_full_input(batch, 0, config, train_data.input_sizes)
                batch_input, batch_target, batch_info = batch

                inp = batch_input[session_idx]  # (L, B, D)
                tar = batch_target[session_idx]  # (seq_length-1, B, D)

                if inp.size(1) == 0:  # Empty batch for this session
                    continue

                # Convert to numpy and collect sequences
                inp_np = inp.cpu().numpy()
                tar_np = tar.cpu().numpy()

                for b in range(inp_np.shape[1]):
                    # Combine context and target to get full sequence
                    context = inp_np[:, b, :]  # (context_length, D)
                    target_end = tar_np[-config.pred_length:, b, :]  # (pred_length, D)
                    full_seq = np.vstack([context, target_end])  # (seq_length, D)
                    session_chunks.append(full_seq)

        # Fit model on collected data
        if len(session_chunks) > 0:
            net.fit_session(session_chunks, session_idx)
            logging.info(f"  Session {session_idx}: fitted on {len(session_chunks)} sequences")

    net.mark_fitted()
    fit_time = time.time() - fit_start_time
    logging.info(f"Fitting completed in {fit_time:.2f}s")

    # Log fitting info
    logger.log_tabular('train/fit_time_seconds', fit_time)
    logger.log_tabular('train/epoch', 0)
    logger.log_tabular('train/batch_number', 0)
    logger.log_tabular('train/samples_seen', 0)
    logger.log_tabular('train/loss', 0.0)

    # Evaluate on validation set
    testloss_list = [float('inf')]
    evaluate_performance(
        net, config, test_data, task_func, logger,
        testloss_list=testloss_list, i_b=0, train_loss=0.0
    )
    logger.dump_tabular()

    # Save fitted model
    torch.save(net.state_dict(), osp.join(config.save_path, 'net_best.pth'))

    total_train_time = time.time() - start_time
    logging.info(f'Total time: {total_train_time:.2f}s')

    log_complete(config.save_path, train_start_time)

    if config.perform_test:
        model_eval(config)


def fomaml_train(config: FOMAMLConfig):
    """
    Meta-training using FOMAML (First-Order MAML).

    This training mode learns a backbone that can quickly adapt to new sessions
    via test-time training on embeddings.
    """
    from poco_ttt.fomaml_trainer import FOMAMLTrainer

    total_train_time = 0.0
    start_time = time.time()

    np.random.seed(NP_SEED + config.seed)
    torch.manual_seed(TCH_SEED + config.seed)
    random.seed(config.seed)
    torch.hub.set_dir(osp.join(DATA_DIR, 'torch_hub'))
    train_start_time = datetime.now()

    if USE_CUDA:
        logging.info("FOMAML training with GPU")

    # Initialize logger with wandb support
    logger = Logger(output_dir=config.save_path, exp_name=config.experiment_name, config=config)

    # Initialize session-level dataset iterators
    train_data = SessionDatasetIters(config, 'train')
    val_data = SessionDatasetIters(config, 'val')

    # Get baseline (copy) MSE for computing val_score
    baseline_mse = val_data.get_baselines(key='session_copy_mse')

    logging.info(f"FOMAML training with {train_data.num_sessions} training sessions")
    logging.info(f"Meta batch size: {config.meta_batch_size}, Inner steps: {config.inner_steps}")

    # Initialize model
    net = model_init(config, train_data.input_sizes, train_data.unit_types)
    param_count = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logging.info(f'Model: {config.model_type}, Total parameters: {param_count}')

    # Get backbone parameters for meta-optimizer
    backbone_params = net.get_backbone_params() if hasattr(net, 'get_backbone_params') else net.parameters()

    # Initialize meta-optimizer (for backbone)
    if config.optimizer_type == 'Adam':
        meta_optimizer = torch.optim.Adam(backbone_params, lr=config.meta_lr, weight_decay=config.wdecay)
    elif config.optimizer_type == 'AdamW':
        meta_optimizer = torch.optim.AdamW(backbone_params, lr=config.meta_lr, weight_decay=config.wdecay, amsgrad=True)
    else:
        meta_optimizer = torch.optim.SGD(backbone_params, lr=config.meta_lr, weight_decay=config.wdecay)

    # Initialize FOMAML trainer
    trainer = FOMAMLTrainer(net, config, meta_optimizer)

    # Learning rate scheduler for meta-optimizer
    if config.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            meta_optimizer,
            T_max=config.max_batch // config.log_every + 1,
            eta_min=config.meta_lr / 20
        )

    # Training loop
    i_b = 0
    val_losses = []
    train_loss_accum = 0.0

    def get_session_data(session_id):
        """Helper function to get support/query data for a session"""
        support_samples, query_samples = train_data.get_support_query_split(session_id)
        if len(support_samples) == 0 or len(query_samples) == 0:
            return None, None
        support_batch = train_data.create_batch_from_data(support_samples, session_id)
        query_batch = train_data.create_batch_from_data(query_samples, session_id)
        return support_batch, query_batch

    logging.info("Starting FOMAML training...")

    while i_b < config.max_batch:
        net.train()

        # Sample meta-batch of sessions
        session_ids = train_data.sample_meta_batch(config.meta_batch_size)

        # Meta-training step
        stats = trainer.meta_train_step(
            session_data_fn=get_session_data,
            session_ids=session_ids,
            pred_length=config.pred_length,
        )

        train_loss_accum += stats['meta_loss']
        i_b += 1

        # Save model periodically
        if i_b % config.save_every == 0:
            torch.save(net.state_dict(), osp.join(config.save_path, f'net_{i_b}.pth'))

        # Log performance
        if i_b % config.log_every == 0:
            logger.log_tabular('train/batch_number', i_b)
            logger.log_tabular('train/meta_loss', train_loss_accum / config.log_every)
            logger.log_tabular('train/inner_loop_loss_avg', stats['avg_inner_loss'])
            logger.log_tabular('train/query_loss_avg', stats['avg_query_loss'])

            # Evaluate on validation set with adaptation
            val_metrics = evaluate_fomaml(net, trainer, val_data, config, baseline_mse=baseline_mse)
            # Loss before adaptation (comparable to standard training)
            logger.log_tabular('val/loss_before_adaptation', val_metrics['pre_adapt_loss'])
            # Loss after adaptation (shows TTT benefit)
            logger.log_tabular('val/loss_after_adaptation', val_metrics['post_adapt_loss'])
            # val_score: 1 - (mse / copy_mse) - comparable metric to baselines
            logger.log_tabular('val/score_before_adaptation', val_metrics['pre_adapt_score'])
            logger.log_tabular('val/score_after_adaptation', val_metrics['post_adapt_score'])
            val_losses.append(val_metrics['post_adapt_loss'])

            # Save best model based on post-adaptation loss
            if val_metrics['post_adapt_loss'] <= min(val_losses):
                torch.save(net.state_dict(), osp.join(config.save_path, 'net_best.pth'))
                logging.info(f"  New best model at step {i_b}, val_loss: {val_metrics['post_adapt_loss']:.6f}")

            logger.dump_tabular()
            train_loss_accum = 0.0

            if config.use_lr_scheduler:
                scheduler.step()

    total_train_time = time.time() - start_time
    logging.info(f'Total FOMAML training time: {total_train_time / 60:.2f} mins')

    # Save final model
    torch.save(net.state_dict(), osp.join(config.save_path, f'net_{i_b}.pth'))

    log_complete(config.save_path, train_start_time)

    if config.perform_test:
        fomaml_eval(config)


def evaluate_fomaml(
    net: nn.Module,
    trainer,
    val_data: SessionDatasetIters,
    config: FOMAMLConfig,
    baseline_mse: list = None,
) -> Dict[str, float]:
    """
    Evaluate FOMAML model on validation set with test-time adaptation.

    For each session:
    1. Split into support/query
    2. Adapt on support
    3. Evaluate on query

    Returns dict with:
    - 'pre_adapt_loss': Average loss before adaptation (comparable to baseline TestLoss)
    - 'post_adapt_loss': Average loss after adaptation (TTT benefit)
    - 'pre_adapt_score': val_score before adaptation (1 - mse/copy_mse)
    - 'post_adapt_score': val_score after adaptation
    """
    net.eval()
    total_pre_loss = 0.0
    total_post_loss = 0.0
    total_pre_score = 0.0
    total_post_score = 0.0
    num_sessions = 0

    for session_id in range(val_data.num_sessions):
        if session_id not in val_data.session_indices:
            continue
        if len(val_data.session_indices[session_id]) < 2:
            continue

        # Get support/query split
        support_samples, query_samples = val_data.get_support_query_split(session_id)
        if len(support_samples) == 0 or len(query_samples) == 0:
            continue

        support_batch = val_data.create_batch_from_data(support_samples, session_id)
        query_batch = val_data.create_batch_from_data(query_samples, session_id)

        if support_batch is None or query_batch is None:
            continue

        # Evaluate with adaptation
        result = trainer.evaluate_with_adaptation(
            support_batch, query_batch, config.pred_length
        )
        total_pre_loss += result['pre_adapt_loss']
        total_post_loss += result['post_adapt_loss']

        # Compute val_score if baseline_mse is provided
        if baseline_mse is not None and session_id < len(baseline_mse):
            copy_mse = baseline_mse[session_id]
            if copy_mse > 0:
                total_pre_score += 1 - result['pre_adapt_loss'] / copy_mse
                total_post_score += 1 - result['post_adapt_loss'] / copy_mse

        num_sessions += 1

    if num_sessions == 0:
        return {
            'pre_adapt_loss': float('inf'),
            'post_adapt_loss': float('inf'),
            'pre_adapt_score': 0.0,
            'post_adapt_score': 0.0,
        }

    return {
        'pre_adapt_loss': total_pre_loss / num_sessions,
        'post_adapt_loss': total_post_loss / num_sessions,
        'pre_adapt_score': total_pre_score / num_sessions if baseline_mse else 0.0,
        'post_adapt_score': total_post_score / num_sessions if baseline_mse else 0.0,
    }


def fomaml_eval(config: FOMAMLConfig):
    """Evaluate FOMAML model on test set with test-time adaptation."""
    from poco_ttt.fomaml_trainer import FOMAMLTrainer

    np.random.seed(NP_SEED)
    torch.manual_seed(TCH_SEED)
    random.seed(config.seed)

    # Load test data
    test_data = SessionDatasetIters(config, 'test')

    # Initialize model and load weights
    net = model_init(config, test_data.input_sizes, test_data.unit_types)
    net.load_state_dict(torch.load(osp.join(config.save_path, 'net_best.pth'), weights_only=True))

    # Create dummy optimizer for trainer (not used during eval)
    dummy_optimizer = torch.optim.SGD(net.parameters(), lr=0.0)
    trainer = FOMAMLTrainer(net, config, dummy_optimizer)

    # Evaluate
    logger = Logger(output_dir=config.save_path, output_fname='test_fomaml.txt', exp_name=config.experiment_name)

    total_pre_adapt = 0.0
    total_post_adapt = 0.0
    num_sessions = 0

    for session_id in range(test_data.num_sessions):
        if session_id not in test_data.session_indices:
            continue
        if len(test_data.session_indices[session_id]) < 2:
            continue

        support_samples, query_samples = test_data.get_support_query_split(session_id)
        if len(support_samples) == 0 or len(query_samples) == 0:
            continue

        support_batch = test_data.create_batch_from_data(support_samples, session_id)
        query_batch = test_data.create_batch_from_data(query_samples, session_id)

        if support_batch is None or query_batch is None:
            continue

        result = trainer.evaluate_with_adaptation(
            support_batch, query_batch, config.pred_length
        )

        total_pre_adapt += result['pre_adapt_loss']
        total_post_adapt += result['post_adapt_loss']
        num_sessions += 1

    if num_sessions > 0:
        avg_pre = total_pre_adapt / num_sessions
        avg_post = total_post_adapt / num_sessions
        improvement = (avg_pre - avg_post) / avg_pre * 100

        logger.log_tabular('test/num_sessions_evaluated', num_sessions)
        logger.log_tabular('test/loss_before_adaptation', avg_pre)
        logger.log_tabular('test/loss_after_adaptation', avg_post)
        logger.log_tabular('test/adaptation_improvement_percent', improvement)
        logger.dump_tabular()

        logging.info(f"FOMAML Test Results:")
        logging.info(f"  Sessions evaluated: {num_sessions}")
        logging.info(f"  Pre-adaptation loss: {avg_pre:.6f}")
        logging.info(f"  Post-adaptation loss: {avg_post:.6f}")
        logging.info(f"  Improvement: {improvement:.2f}%")


def e2e_ttt_train(config):
    """
    Meta-training using E2E-TTT (End-to-End Test-Time Training).

    This training mode uses full second-order gradients (create_graph=True)
    to learn initialization that is optimized for post-adaptation performance.
    """
    from poco_ttt.e2e_ttt_trainer import E2ETTTTrainer

    total_train_time = 0.0
    start_time = time.time()

    np.random.seed(NP_SEED + config.seed)
    torch.manual_seed(TCH_SEED + config.seed)
    random.seed(config.seed)
    torch.hub.set_dir(osp.join(DATA_DIR, 'torch_hub'))
    train_start_time = datetime.now()

    if USE_CUDA:
        logging.info("E2E-TTT training with GPU")

    # Initialize logger with wandb support
    logger = Logger(output_dir=config.save_path, exp_name=config.experiment_name, config=config)

    # Initialize session-level dataset iterators
    train_data = SessionDatasetIters(config, 'train')
    val_data = SessionDatasetIters(config, 'val')

    # Get baseline (copy) MSE for computing val_score
    baseline_mse = val_data.get_baselines(key='session_copy_mse')

    logging.info(f"E2E-TTT training with {train_data.num_sessions} training sessions")
    logging.info(f"Meta batch size: {config.meta_batch_size}, Inner steps: {config.inner_steps}")
    logging.info(f"Using second-order gradients (create_graph=True)")

    # Initialize model
    net = model_init(config, train_data.input_sizes, train_data.unit_types)
    param_count = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logging.info(f'Model: {config.model_type}, Total parameters: {param_count}')

    # For E2E-TTT, we optimize all parameters in the meta-optimizer
    # since gradients flow through the inner loop
    meta_optimizer = torch.optim.Adam(net.parameters(), lr=config.meta_lr, weight_decay=config.wdecay)

    # Initialize E2E-TTT trainer
    trainer = E2ETTTTrainer(net, config, meta_optimizer)

    # Learning rate scheduler for meta-optimizer
    if config.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            meta_optimizer,
            T_max=config.max_batch // config.log_every + 1,
            eta_min=config.meta_lr / 20
        )

    # Training loop
    i_b = 0
    val_losses = []
    train_loss_accum = 0.0

    def get_session_data(session_id):
        """Helper function to get support/query data for a session"""
        support_samples, query_samples = train_data.get_support_query_split(session_id)
        if len(support_samples) == 0 or len(query_samples) == 0:
            return None, None
        support_batch = train_data.create_batch_from_data(support_samples, session_id)
        query_batch = train_data.create_batch_from_data(query_samples, session_id)
        return support_batch, query_batch

    logging.info("Starting E2E-TTT training...")

    while i_b < config.max_batch:
        net.train()

        # Sample meta-batch of sessions
        session_ids = train_data.sample_meta_batch(config.meta_batch_size)

        # Meta-training step with second-order gradients
        stats = trainer.meta_train_step(
            session_data_fn=get_session_data,
            session_ids=session_ids,
            pred_length=config.pred_length,
        )

        train_loss_accum += stats['meta_loss']
        i_b += 1

        # Save model periodically
        if i_b % config.save_every == 0:
            torch.save(net.state_dict(), osp.join(config.save_path, f'net_{i_b}.pth'))

        # Log performance
        if i_b % config.log_every == 0:
            logger.log_tabular('train/batch_number', i_b)
            logger.log_tabular('train/meta_loss', train_loss_accum / config.log_every)
            logger.log_tabular('train/inner_loop_loss_avg', stats['avg_inner_loss'])
            logger.log_tabular('train/query_loss_avg', stats['avg_query_loss'])

            # Evaluate on validation set with adaptation
            val_metrics = evaluate_e2e_ttt(net, trainer, val_data, config, baseline_mse=baseline_mse)
            # Loss before adaptation (comparable to standard training)
            logger.log_tabular('val/loss_before_adaptation', val_metrics['pre_adapt_loss'])
            # Loss after adaptation (shows TTT benefit)
            logger.log_tabular('val/loss_after_adaptation', val_metrics['post_adapt_loss'])
            # val_score: 1 - (mse / copy_mse) - comparable metric to baselines
            logger.log_tabular('val/score_before_adaptation', val_metrics['pre_adapt_score'])
            logger.log_tabular('val/score_after_adaptation', val_metrics['post_adapt_score'])
            val_losses.append(val_metrics['post_adapt_loss'])

            # Save best model based on post-adaptation loss
            if val_metrics['post_adapt_loss'] <= min(val_losses):
                torch.save(net.state_dict(), osp.join(config.save_path, 'net_best.pth'))
                logging.info(f"  New best model at step {i_b}, val_loss: {val_metrics['post_adapt_loss']:.6f}")

            logger.dump_tabular()
            train_loss_accum = 0.0

            if config.use_lr_scheduler:
                scheduler.step()

    total_train_time = time.time() - start_time
    logging.info(f'Total E2E-TTT training time: {total_train_time / 60:.2f} mins')

    # Save final model
    torch.save(net.state_dict(), osp.join(config.save_path, f'net_{i_b}.pth'))

    log_complete(config.save_path, train_start_time)

    if config.perform_test:
        e2e_ttt_eval(config)


def evaluate_e2e_ttt(
    net: nn.Module,
    trainer,
    val_data: SessionDatasetIters,
    config,
    baseline_mse: list = None,
) -> Dict[str, float]:
    """
    Evaluate E2E-TTT model on validation set with test-time adaptation.

    Returns dict with:
    - 'pre_adapt_loss': Average loss before adaptation (comparable to baseline TestLoss)
    - 'post_adapt_loss': Average loss after adaptation (TTT benefit)
    - 'pre_adapt_score': val_score before adaptation (1 - mse/copy_mse)
    - 'post_adapt_score': val_score after adaptation
    """
    net.eval()
    total_pre_loss = 0.0
    total_post_loss = 0.0
    total_pre_score = 0.0
    total_post_score = 0.0
    num_sessions = 0

    for session_id in range(val_data.num_sessions):
        if session_id not in val_data.session_indices:
            continue
        if len(val_data.session_indices[session_id]) < 2:
            continue

        # Get support/query split
        support_samples, query_samples = val_data.get_support_query_split(session_id)
        if len(support_samples) == 0 or len(query_samples) == 0:
            continue

        support_batch = val_data.create_batch_from_data(support_samples, session_id)
        query_batch = val_data.create_batch_from_data(query_samples, session_id)

        if support_batch is None or query_batch is None:
            continue

        # Evaluate with adaptation
        result = trainer.evaluate_with_adaptation(
            support_batch, query_batch, config.pred_length
        )
        total_pre_loss += result['pre_adapt_loss']
        total_post_loss += result['post_adapt_loss']

        # Compute val_score if baseline_mse is provided
        if baseline_mse is not None and session_id < len(baseline_mse):
            copy_mse = baseline_mse[session_id]
            if copy_mse > 0:
                total_pre_score += 1 - result['pre_adapt_loss'] / copy_mse
                total_post_score += 1 - result['post_adapt_loss'] / copy_mse

        num_sessions += 1

    if num_sessions == 0:
        return {
            'pre_adapt_loss': float('inf'),
            'post_adapt_loss': float('inf'),
            'pre_adapt_score': 0.0,
            'post_adapt_score': 0.0,
        }

    return {
        'pre_adapt_loss': total_pre_loss / num_sessions,
        'post_adapt_loss': total_post_loss / num_sessions,
        'pre_adapt_score': total_pre_score / num_sessions if baseline_mse else 0.0,
        'post_adapt_score': total_post_score / num_sessions if baseline_mse else 0.0,
    }


def e2e_ttt_eval(config):
    """Evaluate E2E-TTT model on test set with test-time adaptation."""
    from poco_ttt.e2e_ttt_trainer import E2ETTTTrainer

    np.random.seed(NP_SEED)
    torch.manual_seed(TCH_SEED)
    random.seed(config.seed)

    # Load test data
    test_data = SessionDatasetIters(config, 'test')

    # Initialize model and load weights
    net = model_init(config, test_data.input_sizes, test_data.unit_types)
    net.load_state_dict(torch.load(osp.join(config.save_path, 'net_best.pth'), weights_only=True))

    # Create dummy optimizer for trainer (not used during eval)
    dummy_optimizer = torch.optim.SGD(net.parameters(), lr=0.0)
    trainer = E2ETTTTrainer(net, config, dummy_optimizer)

    # Evaluate
    logger = Logger(output_dir=config.save_path, output_fname='test_e2e_ttt.txt', exp_name=config.experiment_name)

    total_pre_adapt = 0.0
    total_post_adapt = 0.0
    num_sessions = 0

    for session_id in range(test_data.num_sessions):
        if session_id not in test_data.session_indices:
            continue
        if len(test_data.session_indices[session_id]) < 2:
            continue

        support_samples, query_samples = test_data.get_support_query_split(session_id)
        if len(support_samples) == 0 or len(query_samples) == 0:
            continue

        support_batch = test_data.create_batch_from_data(support_samples, session_id)
        query_batch = test_data.create_batch_from_data(query_samples, session_id)

        if support_batch is None or query_batch is None:
            continue

        result = trainer.evaluate_with_adaptation(
            support_batch, query_batch, config.pred_length
        )

        total_pre_adapt += result['pre_adapt_loss']
        total_post_adapt += result['post_adapt_loss']
        num_sessions += 1

    if num_sessions > 0:
        avg_pre = total_pre_adapt / num_sessions
        avg_post = total_post_adapt / num_sessions
        improvement = (avg_pre - avg_post) / avg_pre * 100

        logger.log_tabular('test/num_sessions_evaluated', num_sessions)
        logger.log_tabular('test/loss_before_adaptation', avg_pre)
        logger.log_tabular('test/loss_after_adaptation', avg_post)
        logger.log_tabular('test/adaptation_improvement_percent', improvement)
        logger.dump_tabular()

        logging.info(f"E2E-TTT Test Results:")
        logging.info(f"  Sessions evaluated: {num_sessions}")
        logging.info(f"  Pre-adaptation loss: {avg_pre:.6f}")
        logging.info(f"  Post-adaptation loss: {avg_post:.6f}")
        logging.info(f"  Improvement: {improvement:.2f}%")