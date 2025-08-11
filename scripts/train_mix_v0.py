# train_fixed.py
# 适用于固定相数模型的训练脚本

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.evaluate_mix_spearate_pos_phase_v0 import evaluate_tf as evaluate
from utils.deeplearningutilities.tf import Trainer, MyCheckpointManager
import tensorflow as tf
from datetime import date
import time
from glob import glob
from collections import namedtuple
from datasets.dataset_reader_h5_mix import read_data_train, read_data_val
import numpy as np
import argparse
import yaml

_k = 1000
TrainParams = namedtuple('TrainParams', ['max_iter', 'base_lr', 'batch_size'])
train_params = TrainParams(50 * _k, 0.001, 64)

def create_model(gpu_id=0, **kwargs):
    if gpu_id is not None:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
                print(f"Using GPU {gpu_id}")
            except RuntimeError as e:
                print(f"Error setting up GPU: {e}")

    ## FIXED-MOD: 导入您的固定相数模型
    from models.default_tf_mix_separate_pos_phase_v0 import MultiPhaseParticleNetwork
    
    """返回一个用于训练和评估的网络实例"""
    ## FIXED-MOD: kwargs 现在将传递 'num_phases' 给模型
    model = MultiPhaseParticleNetwork(**kwargs)
    return model


def main():
    parser = argparse.ArgumentParser(description="Training script for Multi-Phase Fluid Network")
    parser.add_argument("cfg", type=str, help="The path to the yaml config file")
    parser.add_argument('--gpu', help='Specify GPU ID (e.g., 0, 1)', type=int, default=0)
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    print(f"Training with config file: {args.cfg}")
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    global train_params
    if 'train_params' in cfg:
        tp_cfg = cfg['train_params']
        train_params = TrainParams(
            tp_cfg.get('max_iter', train_params.max_iter),
            tp_cfg.get('base_lr', train_params.base_lr),
            tp_cfg.get('batch_size', train_params.batch_size)
        )
    print(f"Training Parameters: {train_params}")

    if '2025' not in cfg['train_dir'] and '2026' not in cfg['train_dir']:      # 如果没有指定日期，则使用当前日期
        train_dir_base_name = os.path.splitext(os.path.basename(__file__))[0] + \
                            '_' + os.path.splitext(os.path.basename(args.cfg))[0]
        train_dir = os.path.join(cfg['train_dir'], train_dir_base_name + date.today().strftime("_%Y%m%d"))
    else:
        train_dir = cfg['train_dir']
    print(f"Train directory: {train_dir}")

    val_files = sorted(glob(os.path.join(cfg['dataset_dir'], 'valid', '*.h5')))
    train_files = sorted(glob(os.path.join(cfg['dataset_dir'], 'train', '*.h5')))

    if not train_files:
        sys.exit(f"Error: No training files found in {os.path.join(cfg['dataset_dir'], 'train')}")
    if not val_files:
        print(f"Warning: No validation files found in {os.path.join(cfg['dataset_dir'], 'valid')}")

    val_dataset = read_data_val(files=val_files, window=1, cache_data=True)
    dataset = read_data_train(files=train_files, batch_size=train_params.batch_size, window=3, num_workers=cfg.get('num_workers', 2), **cfg.get('train_data', {}))
    data_iter = iter(dataset)

    trainer = Trainer(train_dir)
    model = create_model(gpu_id=args.gpu, **cfg.get('model', {}))

    try:
        print("Attempting to initialize model for summary...")
        model.init()
    except Exception as e:
        print(f"Could not explicitly initialize model. Error: {e}")

    lr_boundaries = cfg.get('optimizer', {}).get('boundaries', [10*_k, 20*_k, 25*_k, 30*_k, 35*_k])
    lr_values_factors = cfg.get('optimizer', {}).get('lr_value_factors', [1.0, 0.5, 0.25, 0.125, 0.5 * 0.125, 0.25 * 0.125])
    lr_values_actual = [train_params.base_lr * factor for factor in lr_values_factors]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_boundaries, lr_values_actual)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn, epsilon=cfg.get('optimizer', {}).get('epsilon', 1e-6))
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64), model=model, optimizer=optimizer)
    manager = MyCheckpointManager(checkpoint, trainer.checkpoint_dir, keep_checkpoint_steps=list(range(1 * _k, train_params.max_iter + 1, 1 * _k)))

    def euclidean_distance(a, b, epsilon=1e-9):
        a = tf.cast(a, tf.float32)
        b = tf.cast(b, tf.float32)
        return tf.sqrt(tf.reduce_sum(tf.square(a - b), axis=-1) + epsilon)

    def volume_fraction_loss(pr_vol, gt_vol, importance=None, loss_type='mse'):
        pr_vol = tf.cast(pr_vol, tf.float32)
        gt_vol = tf.cast(gt_vol, tf.float32)
        error = tf.reduce_mean(tf.square(pr_vol - gt_vol), axis=-1)
        if importance is not None:
            importance = tf.cast(importance, tf.float32)
            return tf.reduce_mean(importance * error)
        return tf.reduce_mean(error)

    loss_weights = cfg.get('loss_weights', {'pos': 1.0, 'vol': 1.0, 'gamma': 0.5})
    vf_loss_type = cfg.get('loss_vf_type', 'mse')

    def loss_fn(pr_pos, gt_pos, pr_vol=None, gt_vol=None, num_fluid_neighbors=None):
        gamma = tf.cast(loss_weights.get('gamma', 0.5), tf.float32)
        neighbor_scale_val = 1.0 / 40.0
        
        if num_fluid_neighbors is not None and tf.size(num_fluid_neighbors) > 0:
            importance = tf.exp(-neighbor_scale_val * tf.cast(num_fluid_neighbors, tf.float32))
        else:
            dummy_particle_dim_shape = tf.shape(pr_pos)[0]
            importance = tf.ones(shape=(dummy_particle_dim_shape,), dtype=tf.float32)

        pos_loss_val = tf.reduce_mean(importance * tf.pow(euclidean_distance(pr_pos, gt_pos), gamma))
        total_loss = loss_weights.get('pos', 1.0) * pos_loss_val

        if model.num_phases > 1 and pr_vol is not None and gt_vol is not None:
            vol_loss_val = volume_fraction_loss(pr_vol, gt_vol, importance, loss_type=vf_loss_type)
            total_loss += loss_weights.get('vol', 1.0) * vol_loss_val
        
        return total_loss

    @tf.function(experimental_relax_shapes=True)
    def train_step(model_instance, optimizer_instance, current_batch):
        with tf.GradientTape() as tape:
            accumulated_losses = []
            for i in range(train_params.batch_size):
                pos0 = current_batch['pos0'][i]
                vel0 = current_batch['vel0'][i]
                box_pos_sample = current_batch['box'][i]
                box_normals_sample = current_batch['box_normals'][i]
                gt_pos1, gt_pos2 = current_batch['pos1'][i], current_batch['pos2'][i]

                phase_densities_sample = current_batch['density'][i]
                current_vf0 = current_batch.get('phase_fractions0', [None]*train_params.batch_size)[i]
                gt_vf1 = current_batch.get('phase_fractions1', [None]*train_params.batch_size)[i]
                gt_vf2 = current_batch.get('phase_fractions2', [None]*train_params.batch_size)[i]
                cd_val = tf.cast(current_batch.get('cd', [0.5]*train_params.batch_size)[i], tf.float32)
                cf_val = tf.cast(current_batch.get('cf', [0.5]*train_params.batch_size)[i], tf.float32)

                # --- 第一次预测 ---
                inputs1 = (pos0, vel0, current_vf0, box_pos_sample, box_normals_sample)
                pr_pos1, pr_vel1, pr_vf1 = model_instance(inputs1, phase_densities=phase_densities_sample, training=True, cd=cd_val, cf=cf_val)
                loss1 = loss_fn(pr_pos1, gt_pos1, pr_vf1, gt_vf1, getattr(model_instance, 'num_fluid_neighbors', None))
                
                # --- 第二次预测 ---
                inputs2 = (pr_pos1, pr_vel1, pr_vf1, box_pos_sample, box_normals_sample)
                pr_pos2, pr_vel2, pr_vf2 = model_instance(inputs2, phase_densities=phase_densities_sample, training=True, cd=cd_val, cf=cf_val)
                loss2 = loss_fn(pr_pos2, gt_pos2, pr_vf2, gt_vf2, getattr(model_instance, 'num_fluid_neighbors', None))
                
                accumulated_losses.append(0.5 * loss1 + 0.5 * loss2)

            accumulated_losses.extend(model_instance.losses)
            loss_scaling_factor = cfg.get('loss_scaling_factor', 1.0)
            batch_total_loss = loss_scaling_factor * tf.add_n(accumulated_losses) / float(train_params.batch_size)
            grads = tape.gradient(batch_total_loss, model_instance.trainable_variables)
            optimizer_instance.apply_gradients(zip(grads, model_instance.trainable_variables))
        
        return batch_total_loss

    if manager.latest_checkpoint:
        print('restoring from ', manager.latest_checkpoint)
        checkpoint.restore(manager.latest_checkpoint)

    display_str_list = []
    while trainer.keep_training(checkpoint.step, train_params.max_iter, checkpoint_manager=manager, display_str_list=display_str_list):
        data_fetch_start = time.time()
        batch_from_dataset = next(data_iter)
        batch_tf = {}

        for k in ('pos0', 'vel0', 'pos1', 'pos2', 'box', 'box_normals', 'cd', 'cf'):
            if k in batch_from_dataset:
                batch_tf[k] = [tf.convert_to_tensor(x, dtype=tf.float32) for x in batch_from_dataset[k]]
        
        if 'density' in batch_from_dataset:
            batch_tf['density'] = [tf.constant(x, dtype=tf.float32) for x in batch_from_dataset['density']]
        
        for k_vf_idx in range(3):
            k_vf = f'phase_fractions{k_vf_idx}'
            if k_vf in batch_from_dataset:
                processed_vf_list = []
                for vf_sample_np in batch_from_dataset[k_vf]:
                    vf_sample_tf = tf.convert_to_tensor(vf_sample_np, dtype=tf.float32)
                    if model.num_phases > 1 and vf_sample_tf.shape[-1] != model.num_phases:
                        raise ValueError(f"Shape mismatch in {k_vf}: model configured for {model.num_phases} phases, but dataset has {vf_sample_tf.shape[-1]}.")
                    processed_vf_list.append(vf_sample_tf)
                batch_tf[k_vf] = processed_vf_list
        
        data_fetch_latency = time.time() - data_fetch_start
        trainer.log_scalar_every_n_minutes(5, 'DataLatency', data_fetch_latency)

        current_loss = train_step(model, optimizer, batch_tf)
        display_str_list = ['loss', float(current_loss)]

        if trainer.current_step % 10 == 0:
            with trainer.summary_writer.as_default():
                tf.summary.scalar('TotalLoss', current_loss)
                tf.summary.scalar('LearningRate', optimizer.lr(trainer.current_step))

        if trainer.current_step % (1 * _k) == 0:
            if val_files:
                eval_results = evaluate(model, val_dataset, frame_skip=cfg.get('evaluation', {}).get('frame_skip', 20), **cfg.get('evaluation', {}))
                with trainer.summary_writer.as_default():
                    for k_eval, v_eval in eval_results.items():
                        tf.summary.scalar('eval/' + k_eval, v_eval)

    model_weights_name = "model_weights" + date.today().strftime("_%Y_%m_%d") + ".h5"
    model_weights_save_path = os.path.join(train_dir, model_weights_name)
    model.save_weights(model_weights_save_path)
    print(f"Final model weights saved to: {model_weights_save_path}")

    if trainer.current_step >= train_params.max_iter:
        return trainer.STATUS_TRAINING_FINISHED
    else:
        return trainer.STATUS_TRAINING_UNFINISHED

if __name__ == '__main__':
    try:
        import multiprocessing as mp
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    sys.exit(main())