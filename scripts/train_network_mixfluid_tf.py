#!/usr/bin/env python3
import os
import numpy as np
import sys
import argparse
import yaml
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datasets.dataset_reader_h5 import read_data_train, read_data_val
from collections import namedtuple
from glob import glob
import time
from datetime import date
import tensorflow as tf
from utils.deeplearningutilities.tf import Trainer, MyCheckpointManager
# from evaluate_network import evaluate_tf as evaluate
from evaluate_mix_fluid_network import evaluate_tf as evaluate

_k = 1000

TrainParams = namedtuple('TrainParams', ['max_iter', 'base_lr', 'batch_size'])
train_params = TrainParams(50 * _k, 0.001, 16)
# train_params = TrainParams(20 * _k, 0.001, 16)


def create_model(**kwargs):
    from models.default_tf_multi_mix_fluid import MultiPhaseParticleNetwork
    """Returns an instance of the network for training and evaluation"""
    model = MultiPhaseParticleNetwork(**kwargs)
    return model


def main():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("cfg",
                        type=str,
                        help="The path to the yaml config file")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

        # the train dir stores all checkpoints and summaries. The dir name is the name of this file combined with the name of the config file
    train_dir = os.path.splitext(
        os.path.basename(__file__))[0] + '_' + os.path.splitext(
            os.path.basename(args.cfg))[0] + date.today().strftime("_%Y_%m_%d")
    
    train_dir = os.path.join(cfg['train_dir'], train_dir)
    
    print(train_dir) # eg. train_network_tf_6kbox
    print(cfg["train_dir"]) # eg. model_weights.h5

    val_files = sorted(glob(os.path.join(cfg['dataset_dir'], 'valid', '*.h5')))
    train_files = sorted(
        glob(os.path.join(cfg['dataset_dir'], 'train', '*.h5')))

    val_dataset = read_data_val(files=val_files, window=1, cache_data=True)

    dataset = read_data_train(files=train_files,
                              batch_size=train_params.batch_size,
                              window=3,
                              num_workers=2,
                              **cfg.get('train_data', {}))
    data_iter = iter(dataset)

    trainer = Trainer(train_dir)

    model = create_model(**cfg.get('model', {}))

    # boundaries = [
    #     25 * _k,
    #     30 * _k,
    #     35 * _k,
    #     40 * _k,
    #     45 * _k,
    # ]
    boundaries = [
        10 * _k,
        20 * _k,
        25 * _k,
        30 * _k,
        35 * _k,
    ]
    lr_values = [
        train_params.base_lr * 1.0,
        train_params.base_lr * 0.5,
        train_params.base_lr * 0.25,
        train_params.base_lr * 0.125,
        train_params.base_lr * 0.5 * 0.125,
        train_params.base_lr * 0.25 * 0.125,
    ]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, lr_values)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn,
                                         epsilon=1e-6)

    checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                     model=model,
                                     optimizer=optimizer)
    # checkpoint.write('aa')
    # checkpoint.save('aa')

    manager = MyCheckpointManager(checkpoint,
                                  trainer.checkpoint_dir,
                                  keep_checkpoint_steps=list(
                                      range(1 * _k, train_params.max_iter + 1,
                                            1 * _k)))

    def euclidean_distance(a, b, epsilon=1e-9):
        return tf.sqrt(tf.reduce_sum((a - b)**2, axis=-1) + epsilon)
        
    def volume_fraction_loss(pr_vol, gt_vol, importance=None):
        """计算体积分数的损失函数"""
        error = tf.reduce_mean(tf.abs(pr_vol - gt_vol), axis=-1)
        if importance is not None:
            return tf.reduce_mean(importance * error)
        return tf.reduce_mean(error)

    def loss_fn(pr_pos, gt_pos, pr_vol=None, gt_vol=None, num_fluid_neighbors=None):
        """综合损失函数，同时考虑位置误差和体积分数误差"""
        gamma = 0.5
        neighbor_scale = 1 / 40
        importance = tf.exp(-neighbor_scale * num_fluid_neighbors) if num_fluid_neighbors is not None else 1.0
        
        # 位置损失
        pos_loss = tf.reduce_mean(importance * euclidean_distance(pr_pos, gt_pos)**gamma)
        
        # 如果有体积分数数据，添加体积分数损失
        if pr_vol is not None and gt_vol is not None:
            vol_loss = volume_fraction_loss(pr_vol, gt_vol, importance)
            # 体积分数损失权重可以调整
            vol_weight = 0.1
            return pos_loss + vol_weight * vol_loss
        
        return pos_loss

    @tf.function(experimental_relax_shapes=True)
    def train(model, batch):
        with tf.GradientTape() as tape:
            losses = []

            batch_size = train_params.batch_size
            for batch_i in range(batch_size):
                # 准备相体积分数数据，如果存在的话
                phase_fractions = None
                if 'phase_fractions0' in batch:
                    phase_fractions = batch['phase_fractions0'][batch_i]
                
                # 准备扩散和交换系数 - 确保是标量值
                cd = tf.constant(0.5, dtype=tf.float32)
                cf = tf.constant(0.5, dtype=tf.float32)
                
                if 'cd' in batch:
                    # 获取本批次的标量值
                    cd_value = batch['cd'] if isinstance(batch['cd'], float) else batch['cd'][batch_i]
                    cd = tf.cast(cd_value, dtype=tf.float32)
                if 'cf' in batch:
                    # 获取本批次的标量值
                    cf_value = batch['cf'] if isinstance(batch['cf'], float) else batch['cf'][batch_i]
                    cf = tf.cast(cf_value, dtype=tf.float32)
                
                # 第一帧预测
                inputs = ([
                    batch['pos0'][batch_i], batch['vel0'][batch_i], phase_fractions,
                    batch['box'][batch_i], batch['box_normals'][batch_i]
                ])
                
                # # 打印调试信息，检查输入特征形状
                # print("Input pos shape:", tf.shape(batch['pos0'][batch_i]))
                # if phase_fractions is not None:
                #     print("Input phase shape:", tf.shape(phase_fractions))

                if model.num_phases > 1 and phase_fractions is not None:
                    pr_pos1, pr_vel1, pr_phase1 = model(inputs, cd=cd, cf=cf)
                    
                    # 计算位置和体积分数损失
                    gt_phase1 = batch['phase_fractions1'][batch_i] if 'phase_fractions1' in batch else phase_fractions
                    l = 0.5 * loss_fn(pr_pos1, batch['pos1'][batch_i],
                                     pr_phase1, gt_phase1,
                                     model.num_fluid_neighbors)
                    
                    # 第二帧预测
                    inputs = (pr_pos1, pr_vel1, pr_phase1, batch['box'][batch_i],
                              batch['box_normals'][batch_i])
                    pr_pos2, pr_vel2, pr_phase2 = model(inputs, cd=cd, cf=cf)
                    
                    # 计算第二帧损失
                    gt_phase2 = batch['phase_fractions2'][batch_i] if 'phase_fractions2' in batch else gt_phase1
                    l += 0.5 * loss_fn(pr_pos2, batch['pos2'][batch_i],
                                      pr_phase2, gt_phase2,
                                      model.num_fluid_neighbors)
                else:
                    # 单相流体处理逻辑，保持原有
                    pr_pos1, pr_vel1 = model(inputs)
                    l = 0.5 * loss_fn(pr_pos1, batch['pos1'][batch_i],
                                      None, None,
                                      model.num_fluid_neighbors)

                    inputs = (pr_pos1, pr_vel1, None, batch['box'][batch_i],
                              batch['box_normals'][batch_i])
                    pr_pos2, pr_vel2 = model(inputs)
                    l += 0.5 * loss_fn(pr_pos2, batch['pos2'][batch_i],
                                       None, None,
                                       model.num_fluid_neighbors)
                
                losses.append(l)

            losses.extend(model.losses)
            total_loss = 128 * tf.add_n(losses) / batch_size

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss

    if manager.latest_checkpoint:
        print('restoring from ', manager.latest_checkpoint)
        checkpoint.restore(manager.latest_checkpoint)

    display_str_list = []
    while trainer.keep_training(checkpoint.step,
                                train_params.max_iter,
                                checkpoint_manager=manager,
                                display_str_list=display_str_list):

        data_fetch_start = time.time()
        # 从数据集中获取一个批次
        # batch["pos0"]: [pos] * batch_size
        # batch["pos1"]: [pos] * batch_size
        # batch["pos2"]: [pos] * batch_size
        # batch["vel0"]: [vel] * batch_size
        # batch["box"]: [box] * batch_size
        # batch["box_normals"]: [box_normals] * batch_size
        batch = next(data_iter)
        batch_tf = {}
        for k in ('pos0', 'vel0', 'pos1', 'pos2', 'box', 'box_normals'):
            if k in batch:
                batch_tf[k] = [tf.convert_to_tensor(x) for x in batch[k]]
                
        # 添加多相流体的体积分数数据处理
        for k in ('phase_fractions0', 'phase_fractions1', 'phase_fractions2'):
            if k in batch:
                batch_tf[k] = [tf.convert_to_tensor(x) for x in batch[k]]
                
        # 添加混合系数 - 修改为正确处理批次中的 cd 和 cf
        if 'cd' in batch:
            # 从批次中获取的 cd 是列表，需要获取对应批次的值并转为标量
            batch_tf['cd'] = tf.convert_to_tensor(batch['cd'], dtype=tf.float32)
        if 'cf' in batch:
            # 从批次中获取的 cf 是列表，需要获取对应批次的值并转为标量
            batch_tf['cf'] = tf.convert_to_tensor(batch['cf'], dtype=tf.float32)
        # print("Batch cd :", batch['cd'])
        # print("Batch cd :", batch_tf['cd'])
            
        data_fetch_latency = time.time() - data_fetch_start
        trainer.log_scalar_every_n_minutes(5, 'DataLatency', data_fetch_latency)

        current_loss = train(model, batch_tf)
        display_str_list = ['loss', float(current_loss)]

        if trainer.current_step % 10 == 0:
            with trainer.summary_writer.as_default():
                tf.summary.scalar('TotalLoss', current_loss)
                tf.summary.scalar('LearningRate',
                                  optimizer.lr(trainer.current_step))

        if trainer.current_step % (1 * _k) == 0:
            for k, v in evaluate(model,
                                 val_dataset,
                                 frame_skip=20,
                                 **cfg.get('evaluation', {})).items():
                with trainer.summary_writer.as_default():
                    tf.summary.scalar('eval/' + k, v)

    model_weights_save_path = "model_weights" + date.today().strftime("_%Y_%m_%d") + ".h5"
    model_weights_save_path = os.path.join(train_dir, model_weights_save_path)
    model.save_weights(model_weights_save_path)

    if trainer.current_step == train_params.max_iter:
        return trainer.STATUS_TRAINING_FINISHED
    else:
        return trainer.STATUS_TRAINING_UNFINISHED


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn')
    sys.exit(main())
