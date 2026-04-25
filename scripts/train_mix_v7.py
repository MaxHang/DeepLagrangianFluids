#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Assuming evaluate_mix_network is adapted for the new model output
# 假设 evaluate_mix_network 已经适配了新的模型输出
from scripts.evaluate_mix_spearate_pos_phase_v1 import evaluate_tf as evaluate
from utils.deeplearningutilities.tf import Trainer, MyCheckpointManager
# from utils.deeplearningutilities.tf import Trainer, MyCheckpointManagerWithCache
from models.default_tf_mix_separate_pos_phase_v7 import MultiPhaseParticleNetwork
import tensorflow as tf
from datetime import date, datetime
import time
from glob import glob
from collections import namedtuple
# Ensure these functions can return phase_fractions and handle num_phases
# 确保这些函数能返回 phase_fractions 并处理 num_phases
from datasets.dataset_reader_h5_mix import read_data_train, read_data_val
import numpy as np
import argparse
import yaml
import shutil

tf.debugging.enable_check_numerics()

_k = 1000
TrainParams = namedtuple('TrainParams', ['max_iter', 'base_lr', 'batch_size'])
# Default values, can be overridden by cfg
# 默认值，可以被cfg覆盖
train_params = TrainParams(100 * _k, 0.001, 32)


def create_model(gpu_id=0, **kwargs) -> MultiPhaseParticleNetwork: # Receives model_config # 接收 model_config
    if gpu_id is not None:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # --- [核心修改] ---
            # 检查传入的 gpu_id 是否在 TensorFlow 可见的 GPU 列表范围内
            if gpu_id >= len(gpus):
                print(f"[WARNING] Requested GPU ID {gpu_id} is out of range. TensorFlow sees {len(gpus)} GPU(s).")
                print(f"[WARNING] This might be because CUDA_VISIBLE_DEVICES is set.")
                print(f"[WARNING] Defaulting to the first available GPU: {gpus[0].name}")
                # 默认使用列表中的第一个 (也就是唯一可见的那个)
                target_gpu = gpus[0]
            else:
                target_gpu = gpus[gpu_id]
            # --- [修改结束] ---
            try:
                tf.config.set_visible_devices(target_gpu, 'GPU')
                tf.config.experimental.set_memory_growth(target_gpu, True)
                print(f"Using GPU {gpu_id}")
            except RuntimeError as e:
                print(f"Error setting up GPU: {e}")
                # Fallback or exit if GPU setup fails
                # 如果GPU设置失败，则回退或退出
                # For now, just print and continue (hoping CPU works or user notices)
                # 目前，仅打印并继续（希望CPU能工作或用户注意到）
    # Ensure the import path is correct
    # 确保导入路径正确
    """Returns an instance of the network for training and evaluation"""
    model = MultiPhaseParticleNetwork(**kwargs)
    return model

def main():
    parser = argparse.ArgumentParser(description="Training script for Multi-Phase Fluid Network")
    parser.add_argument("cfg", type=str, help="The path to the yaml config file")
    # Changed to int
    parser.add_argument('--gpu', help='Specify GPU ID (e.g., 0, 1)', type=int, default=0)
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    print(f"Training with config file: {args.cfg}")
    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    print(f"Dataset directory: {cfg['dataset_dir']}")

    global train_params, _k  # 👈 就加这一行！

    tp_cfg = cfg.get('train_params', {})
    train_params = train_params._replace(**tp_cfg)
    
    print(f"Training parameters: max_iter={train_params.max_iter}, base_lr={train_params.base_lr}, batch_size={train_params.batch_size}")

    _k = train_params.max_iter // 50
    print(f"Calculated _k for checkpointing: {_k}")

    if '2025' not in cfg['train_dir'] and '2026' not in cfg['train_dir']:      # 如果没有指定日期，则使用当前日期
        train_dir = os.path.join(cfg['train_dir'], datetime.now().strftime("%Y%m%d%H%M%S"))
        os.makedirs(train_dir, exist_ok=True)
        try:
            cfg['train_dir'] = train_dir
            # 写入到新目录的配置文件
            target_cfg_path = os.path.join(train_dir, 'training_config.yaml')
            with open(target_cfg_path, 'w', encoding='utf-8') as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        except Exception as e:
            print(f"copy faild : {str(e)}")
    else:
        train_dir = cfg['train_dir']

    print(f"Train directory: {train_dir}")

    val_files = sorted(glob(os.path.join(cfg['dataset_dir'], 'valid', '*.h5')))
    train_files = sorted(glob(os.path.join(cfg['dataset_dir'], 'train', '*.h5')))

    if not train_files:
        print(f"Error: No training files found in {os.path.join(cfg['dataset_dir'], 'train')}")
        sys.exit(1)
    if not val_files:
        print(f"Warning: No validation files found in {os.path.join(cfg['dataset_dir'], 'valid')}")


    val_dataset = read_data_val(files=val_files, window=1, cache_data=True)

    print(cfg.get('train_data'))

    dataset = read_data_train(files=train_files,
                              batch_size=train_params.batch_size,
                              window=3, # For 2-step prediction 
                              num_workers=cfg.get('num_workers', 2),
                              **cfg.get('train_data', {}))
    data_iter = iter(dataset)

    trainer = Trainer(train_dir)
    # Get model config from YAML
    model = create_model(gpu_id=args.gpu, **cfg.get('model', {}))

    # save_complete_config(cfg, train_dir, train_params, model.init_params)

    # try:
    #     print("Attempting to initialize model for summary...")
    #     # Example values
    #     model.init()
    # except Exception as e:
    #     print(f"Could not explicitly initialize model, will build on first data pass. Error: {e}")

    # Learning rate schedule
    lr_boundaries = cfg.get('optimizer', {}).get('boundaries', [10*_k, 20*_k, 25*_k, 30*_k, 35*_k])
    lr_values_factors = cfg.get('optimizer', {}).get('lr_value_factors', [1.0, 0.5, 0.25, 0.125, 0.5 * 0.125, 0.25 * 0.125])
    lr_values_actual = [train_params.base_lr * factor for factor in lr_values_factors]
    optimizer_epsilon = cfg.get('optimizer', {}).get('epsilon', 1e-6)

    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        lr_boundaries, lr_values_actual)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_fn,
        clipnorm=1.0,
        epsilon=optimizer_epsilon
    )

    # Get loss weights from config
    loss_weights = cfg.get(
        'loss_weights', {
            'pos': 1.0, 
            'vol': 1.0, 
            'gamma': 0.5,
            'use_mass_conservation': False,
            'mass_conservation': 0.1,
        }
    )

    vf_loss_type = cfg.get('loss_vf_type', 'mse')
    use_mass_conservation = loss_weights.get('use_mass_conservation', False)
    mass_conservation_weight = loss_weights.get('mass_conservation', 0.1)
    num_workers = cfg.get('num_workers', 2)
    loss_scaling_factor = cfg.get('loss_scaling_factor', 1.0)

    # Ensure step is int64
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64),
                                     model=model,
                                     optimizer=optimizer)

    # manager = MyCheckpointManagerWithCache(
    #     checkpoint=checkpoint,
    #     directory=trainer.checkpoint_dir,  # NAS 目录
    #     keep_checkpoint_steps=list(range(1 * _k, train_params.max_iter + 1, 1 * _k)),
    #     local_cache_dir="/tmp/deeplearning_cache",  # 本地缓存目录
    #     max_local_cache=3,  # 本地保留最近3个
    #     upload_timeout=300  # 上传超时5分钟
    # )
    
    manager = MyCheckpointManager(checkpoint,
                                  trainer.checkpoint_dir,
                                  keep_checkpoint_steps=list(
                                      range(1 * _k, train_params.max_iter + 1,
                                            1 * _k)))


    def total_mass_conservation_loss(vf_next, vf_current, phase_densities):
        """
        全局单相总质量守恒损失
        约束: M_phase_i(t+1) = M_phase_i(t) = Σ_particles[VF_j,i × ρ_mix_j]

        这是唯一需要的物理约束，因为：
        1. VF的MSE损失已经约束单粒子的VF准确性
        2. 模型的normalization层已经保证 Σ VF = 1
        3. 但以上都不能保证全局单相总质量守恒
        """
        # 计算每个粒子的混合密度
        rho_mix_current = tf.reduce_sum(vf_current * phase_densities, axis=-1, keepdims=True)  # [N, 1]
        rho_mix_next = tf.reduce_sum(vf_next * phase_densities, axis=-1, keepdims=True)  # [N, 1]
        
        # 计算每相的总质量
        mass_per_phase_current = vf_current * rho_mix_current  # [N, num_phases]
        mass_per_phase_next = vf_next * rho_mix_next  # [N, num_phases]
        
        total_mass_current = tf.reduce_sum(mass_per_phase_current, axis=0)  # [num_phases]
        total_mass_next = tf.reduce_sum(mass_per_phase_next, axis=0)  # [num_phases]
        
        # 相对误差
        mass_drift_per_phase = tf.abs(total_mass_next - total_mass_current) / (total_mass_current + 1e-8)
        
        return tf.reduce_mean(mass_drift_per_phase)

    def euclidean_distance(a, b, epsilon=1e-9):
        # Ensure a and b are float32 for stability with sqrt
        a = tf.cast(a, tf.float32)
        b = tf.cast(b, tf.float32)
        return tf.sqrt(tf.reduce_sum(tf.square(a - b), axis=-1) + epsilon)

    # Allow choosing loss type
    def volume_fraction_loss(pr_vol, gt_vol, current_num_phases, importance=None, loss_type='mse'):
            pr_vol = tf.cast(pr_vol, tf.float32)
            gt_vol = tf.cast(gt_vol, tf.float32)

            error = tf.reduce_mean(tf.square(pr_vol - gt_vol), axis=-1)
            
            if importance is not None:
                importance = tf.cast(importance, tf.float32)
                return tf.reduce_mean(importance * error)
            return tf.reduce_mean(error)

    # e.g., 'mse', 'kl_divergence', 'combined'
    vf_loss_type = cfg.get('loss_vf_type', 'mse')
    use_mass_conservation = loss_weights.get('use_mass_conservation', False)
    mass_conservation_weight = loss_weights.get('mass_conservation', 0.1)

    def loss_fn(pr_pos, gt_pos, pr_vol, gt_vol, current_vol, phase_densities,
                current_num_phases, num_fluid_neighbors=None):
        """
        Args:
            pr_pos: 预测位置 [N, 3]
            gt_pos: 真实位置 [N, 3]
            pr_vol: 预测体积分数 [N, num_phases]
            gt_vol: 真实体积分数 [N, num_phases]
            current_vol: 当前体积分数 [N, num_phases]  # 新增!
            phase_densities: 各相密度 [num_phases]      # 新增!
            current_num_phases: 当前相数
            num_fluid_neighbors: 邻居数量
        """
        gamma = tf.cast(loss_weights.get('gamma', 0.5), tf.float32)
        # Default neighbor_scale if num_fluid_neighbors is None or not effective
        neighbor_scale_val = 1.0 / 40.0
        
        if num_fluid_neighbors is not None and tf.size(num_fluid_neighbors) > 0 :
            importance = tf.exp(-neighbor_scale_val * tf.cast(num_fluid_neighbors, tf.float32))
        else:
            dummy_particle_dim_shape = tf.shape(pr_pos)[0] # Assuming pr_pos is [particles_in_sample, 3] # 假设pr_pos是[样本中的粒子数, 3]
            importance = tf.ones(shape=(dummy_particle_dim_shape,), dtype=tf.float32)

        # 位置损失
        pos_loss_val = tf.reduce_mean(importance * tf.pow(euclidean_distance(pr_pos, gt_pos), gamma))
        total_loss = loss_weights.get('pos', 1.0) * pos_loss_val

        # VF 损失（MSE）
        if pr_vol is not None and gt_vol is not None:
            # For now, assume gt_vol is also [particles, num_phases]
            vol_loss_val = volume_fraction_loss(pr_vol, gt_vol, current_num_phases, importance, loss_type=vf_loss_type)
            total_loss += loss_weights.get('vol', 1.0) * vol_loss_val

        # ========== 全局单相总质量守恒约束 ==========
        if use_mass_conservation and pr_vol is not None and current_vol is not None and phase_densities is not None:
            mass_loss = total_mass_conservation_loss(pr_vol, current_vol, phase_densities)
            total_loss += mass_conservation_weight * mass_loss

        return total_loss


    @tf.function(experimental_relax_shapes=True)
    # Renamed for clarity
    # 为清晰起见重命名
    def train_step(model_instance, optimizer_instance, current_batch):
        with tf.GradientTape() as tape:
            # Accumulate loss for each item in the batch
            accumulated_losses = []

            # Iterate over each sample in the batch (as loaded by dataset reader)
            for i in range(train_params.batch_size):
                pos0 = current_batch['pos0'][i]
                vel0 = current_batch['vel0'][i]
                box_pos_sample = current_batch['box'][i]
                box_normals_sample = current_batch['box_normals'][i]

                gt_pos1 = current_batch['pos1'][i]
                # For 2-step prediction
                gt_pos2 = current_batch['pos2'][i]

                current_num_phases_sample = current_batch['num_phases'][i]
                phase_densities_sample = current_batch['density'][i]
                current_vf0 = current_batch.get['phase_fractions0'][i]
                gt_vf1 = current_batch.get['phase_fractions1'][i]
                gt_vf2 = current_batch.get['phase_fractions2'][i]

                cd_val = tf.cast(current_batch.get['cd'][i], tf.float32)
                cf_val = tf.cast(current_batch.get['cf'][i], tf.float32)

                # --- First prediction step ---
                inputs1 = (pos0, vel0, current_vf0, box_pos_sample, box_normals_sample)
                pr_pos1, pr_vel1, pr_vf1 = model_instance(
                    inputs1, 
                    current_num_phases = current_num_phases_sample,
                    phase_densities=phase_densities_sample,
                    training=True, cd=cd_val, cf=cf_val
                )
                
                loss1 = loss_fn(pr_pos1, gt_pos1, pr_vf1, gt_vf1, current_vf0, phase_densities_sample,
                                current_num_phases_sample, model_instance.num_fluid_neighbors)
                
                # --- Second prediction step ---
                # Use predicted as input
                inputs2 = (pr_pos1, pr_vel1, pr_vf1, box_pos_sample, box_normals_sample)
                pr_pos2, pr_vel2, pr_vf2 = model_instance(
                    inputs2, 
                    current_num_phases=current_num_phases_sample,
                    phase_densities=phase_densities_sample,
                    training=True, cd=cd_val, cf=cf_val)

                loss2 = loss_fn(pr_pos2, gt_pos2, pr_vf2, gt_vf2, pr_vf1, phase_densities_sample,
                                current_num_phases_sample, model_instance.num_fluid_neighbors)
                
                accumulated_losses.append(0.5 * loss1 + 0.5 * loss2)

            accumulated_losses.extend(model_instance.losses)

            # Average loss over the batch
            # Loss scaling factor, can be tuned or made a config parameter
            loss_scaling_factor = cfg.get('loss_scaling_factor', 1.0)
            batch_total_loss = loss_scaling_factor * tf.add_n(accumulated_losses) / float(train_params.batch_size)

            grads = tape.gradient(batch_total_loss, model_instance.trainable_variables)
            optimizer_instance.apply_gradients(zip(grads, model_instance.trainable_variables))
        
        return batch_total_loss


    if manager.latest_checkpoint:
        print('restoring from ', manager.latest_checkpoint)
        checkpoint.restore(manager.latest_checkpoint)


    display_str_list = []
    # Main training loop
    while trainer.keep_training(checkpoint.step,
                                train_params.max_iter,
                                checkpoint_manager=manager,
                                display_str_list=display_str_list):
        data_fetch_start = time.time()
        batch_from_dataset = next(data_iter)
        
        # Convert numpy arrays from dataset to TensorFlow tensors
        batch_tf = {}
        # Standard features
        for k in ('pos0', 'vel0', 'pos1', 'pos2', 'box', 'box_normals'):
            if k in batch_from_dataset:
                batch_tf[k] = [tf.convert_to_tensor(x, dtype=tf.float32) for x in batch_from_dataset[k]]

        if 'num_phases' in batch_from_dataset:
            batch_tf['num_phases'] = [tf.constant(x, dtype=tf.int32) for x in batch_from_dataset['num_phases']]
        if 'density' in batch_from_dataset:
            batch_tf['density'] = [tf.constant(x, dtype=tf.float32) for x in batch_from_dataset['density']]
        
        for k_vf_idx in range(3): # For phase_fractions0, 1, 2
            k_vf = f'phase_fractions{k_vf_idx}'
            if k_vf in batch_from_dataset:
                processed_vf_list = []
                for i, vf_sample_np in enumerate(batch_from_dataset[k_vf]):
                    vf_sample_tf = tf.convert_to_tensor(vf_sample_np, dtype=tf.float32)
                    num_phases_for_sample = batch_from_dataset['num_phases'][i]
                    if vf_sample_tf.shape[-1] != num_phases_for_sample:
                        raise ValueError(
                            f"Shape mismatch for sample {i} in {k_vf}: "
                            f"dataset claimed {num_phases_for_sample} phases, "
                            f"but phase_fractions tensor has shape {vf_sample_tf.shape}. "
                            "Please ensure your dataset reader provides the correct number of phases and "
                            "the full phase fraction data for each sample."
                        )
                    processed_vf_list.append(vf_sample_tf)
                batch_tf[k_vf] = processed_vf_list
        
        # Cd and Cf - assuming they are single scalar values per batch or lists of scalars
        if 'cd' in batch_from_dataset:
            # batch_tf['cd'] will be used by train_step which expects a scalar or list of scalars
            batch_tf['cd'] = batch_from_dataset['cd']
        if 'cf' in batch_from_dataset:
            batch_tf['cf'] = batch_from_dataset['cf']

        data_fetch_latency = time.time() - data_fetch_start
        trainer.log_scalar_every_n_minutes(5, 'DataLatency', data_fetch_latency)

        current_loss = train_step(model, optimizer, batch_tf)
        # Update display string
        display_str_list = ['loss', float(current_loss)]

        if trainer.current_step % 10 == 0:
            with trainer.summary_writer.as_default():
                tf.summary.scalar('TotalLoss', current_loss)
                tf.summary.scalar('LearningRate', optimizer.lr(trainer.current_step))

        if trainer.current_step > 100 and trainer.current_step % (1 * _k) == 0:
            # Only evaluate if validation files exist
            if val_files:
                eval_results = evaluate(model,
                                        val_dataset, # val_dataset reader should also handle num_phases 
                                        frame_skip=cfg.get('evaluation', {}).get('frame_skip', 20),
                                        **cfg.get('evaluation', {}))
                with trainer.summary_writer.as_default():
                    for k_eval, v_eval in eval_results.items():
                        tf.summary.scalar('eval/' + k_eval, v_eval)
            else:
                print(f"Step {trainer.current_step}: Skipping evaluation as no validation files are present.")


    # 等待所有待上传任务完成  manager cache version
    # manager.wait_for_pending_uploads(timeout=600)

    model_weights_name = "model_weights" + \
        date.today().strftime("_%Y_%m_%d") + ".h5"
    model_weights_save_path = os.path.join(train_dir, model_weights_name)
    model.save_weights(model_weights_save_path)
    print(f"Final model weights saved to: {model_weights_save_path}")

    if trainer.current_step >= train_params.max_iter:
        print("Training finished.")
        return trainer.STATUS_TRAINING_FINISHED
    else:
        print("Training stopped before max_iter.")
        return trainer.STATUS_TRAINING_UNFINISHED


if __name__ == '__main__':
    # It's good practice to set this for TensorFlow if using multiprocessing,
    # though 'spawn' is often default on non-Linux.
    # 'fork' can cause issues with CUDA in child processes.
    try:
        # force=True if already set
        import multiprocessing as mp
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        print("Multiprocessing start method already set or 'spawn' not supported.")

    sys.exit(main())