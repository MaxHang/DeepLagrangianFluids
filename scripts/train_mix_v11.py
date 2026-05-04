#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
import numpy as np
import argparse, yaml, time
from glob import glob
from datetime import datetime, date
from collections import namedtuple

from utils.deeplearningutilities.tf import Trainer, MyCheckpointManager
from models.default_tf_mix_separate_pos_phase_v11 import MultiPhaseParticleNetwork
from datasets.dataset_reader_h5_mix import read_data_train, read_data_val
from scripts.evaluate_mix_spearate_pos_phase_v1 import evaluate_tf as evaluate

tf.debugging.enable_check_numerics()

# ===========================
# 训练参数
# ===========================
_k = 1000
TrainParams = namedtuple('TrainParams', ['max_iter', 'base_lr', 'batch_size'])
train_params = TrainParams(50000, 0.001, 32)


# ===========================
# GPU + Model
# ===========================
def create_model(gpu_id=0, **kwargs):
    """
    创建模型，并绑定GPU
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        target_gpu = gpus[min(gpu_id, len(gpus)-1)]
        tf.config.set_visible_devices(target_gpu, 'GPU')
        tf.config.experimental.set_memory_growth(target_gpu, True)
        print(f"[INFO] Using GPU: {target_gpu.name}")
    else:
        print("[INFO] No GPU found, using CPU")

    model = MultiPhaseParticleNetwork(**kwargs)
    return model


# ===========================
# Loss Functions（核心修改）
# ===========================

def kl_vf_loss(pr, gt, importance=None):
    """
    KL 散度损失（替代 MSE）
    作用：
      - 强烈惩罚“GT=0但预测>0”（解决凭空生成相）
      - 不鼓励均匀分布（解决1:1:1问题）
    """
    pr = tf.clip_by_value(pr, 1e-6, 1.0)
    gt = tf.clip_by_value(gt, 1e-6, 1.0)

    kl = tf.reduce_sum(gt * tf.math.log(gt / pr), axis=-1)

    if importance is not None:
        return tf.reduce_mean(importance * kl)
    return tf.reduce_mean(kl)


def zero_phase_penalty(pr, gt):
    """
    防止生成不存在的相（核心约束）
    GT=0 的相 → prediction 必须接近0
    """
    mask = tf.cast(gt < 1e-6, tf.float32)
    return tf.reduce_mean(mask * pr)


def entropy_loss(pr):
    """
    熵损失（抑制均匀分布）
    熵越高 → 越接近平均分布 → 惩罚
    """
    ent = -tf.reduce_sum(pr * tf.math.log(pr + 1e-8), axis=-1)
    return tf.reduce_mean(ent)


def total_mass_conservation_loss(vf_next, vf_current, phase_densities):
    """
    全局质量守恒（极其关键）
    防止整体比例漂移 → 避免收敛到1:1:1
    """
    rho_cur = tf.reduce_sum(vf_current * phase_densities, axis=-1, keepdims=True)
    rho_next = tf.reduce_sum(vf_next * phase_densities, axis=-1, keepdims=True)

    mass_cur = vf_current * rho_cur
    mass_next = vf_next * rho_next

    total_cur = tf.reduce_sum(mass_cur, axis=0)
    total_next = tf.reduce_sum(mass_next, axis=0)

    drift = tf.abs(total_next - total_cur) / (total_cur + 1e-8)
    return tf.reduce_mean(drift)


def loss_fn(pr_pos, gt_pos, pr_vf, gt_vf, cur_vf, densities):
    """
    总损失函数（组合多个约束）
    """

    # --- 位置损失 ---
    pos_err = tf.sqrt(tf.reduce_sum((pr_pos - gt_pos)**2, axis=-1) + 1e-9)
    pos_loss = tf.reduce_mean(pos_err)

    # --- VF KL损失 ---
    vf_loss = kl_vf_loss(pr_vf, gt_vf)

    # --- 防止生成不存在相 ---
    zero_loss = zero_phase_penalty(pr_vf, gt_vf)

    # --- 抑制均匀分布 ---
    ent_loss = entropy_loss(pr_vf)

    # --- 全局质量守恒 ---
    mass_loss = total_mass_conservation_loss(pr_vf, cur_vf, densities)

    total = (
        1.0 * pos_loss +
        5.0 * vf_loss +
        2.0 * zero_loss +
        0.01 * ent_loss +
        1.0 * mass_loss
    )

    return total, pos_loss, vf_loss, zero_loss, ent_loss, mass_loss


# ===========================
# Train Step
# ===========================

@tf.function(experimental_relax_shapes=True)
def train_step(model, optimizer, batch):

    with tf.GradientTape() as tape:
        losses = []

        # 记录子loss（用于打印）
        pos_l, vf_l, zero_l, ent_l, mass_l = 0., 0., 0., 0., 0.

        for i in range(train_params.batch_size):
            pos0 = batch['pos0'][i]
            vel0 = batch['vel0'][i]
            box = batch['box'][i]
            box_n = batch['box_normals'][i]

            gt_pos1 = batch['pos1'][i]
            gt_pos2 = batch['pos2'][i]

            vf0 = batch['phase_fractions0'][i]
            vf1 = batch['phase_fractions1'][i]
            vf2 = batch['phase_fractions2'][i]

            dens = batch['density'][i]

            cd = tf.cast(batch['cd'][i], tf.float32)
            cf = tf.cast(batch['cf'][i], tf.float32)

            # ---- step 1 ----
            p1, v1, pred_vf1 = model(
                (pos0, vel0, vf0, box, box_n),
                phase_densities=dens,
                training=True, cd=cd, cf=cf
            )

            l1, p_l1, vf_l1, z_l1, e_l1, m_l1 = loss_fn(p1, gt_pos1, pred_vf1, vf1, vf0, dens)

            # ---- step 2 ----
            p2, v2, pred_vf2 = model(
                (p1, v1, pred_vf1, box, box_n),
                phase_densities=dens,
                training=True, cd=cd, cf=cf
            )

            l2, p_l2, vf_l2, z_l2, e_l2, m_l2 = loss_fn(p2, gt_pos2, pred_vf2, vf2, pred_vf1, dens)

            losses.append(0.5 * (l1 + l2))

            pos_l += p_l1 + p_l2
            vf_l += vf_l1 + vf_l2
            zero_l += z_l1 + z_l2
            ent_l += e_l1 + e_l2
            mass_l += m_l1 + m_l2

        total_loss = tf.add_n(losses) / float(train_params.batch_size)

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return total_loss, pos_l, vf_l, zero_l, ent_l, mass_l


# ===========================
# Main
# ===========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    print(f"[INFO] Loading config: {args.cfg}")
    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    global train_params, _k
    train_params = train_params._replace(**cfg.get('train_params', {}))
    _k = train_params.max_iter // 50

    print(f"[INFO] Training params: {train_params}")

    # ===== 自动创建 train_dir 并写入 cfg =====
    if '2025' not in cfg['train_dir'] and '2026' not in cfg['train_dir']:
        train_dir = os.path.join(cfg['train_dir'], datetime.now().strftime("%Y%m%d%H%M%S"))
        os.makedirs(train_dir, exist_ok=True)

        cfg['train_dir'] = train_dir
        with open(os.path.join(train_dir, 'training_config.yaml'), 'w') as f:
            yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)

    else:
        train_dir = cfg['train_dir']

    print(f"[INFO] Train directory: {train_dir}")

    train_files = sorted(glob(os.path.join(cfg['dataset_dir'], 'train', '*.h5')))
    val_files = sorted(glob(os.path.join(cfg['dataset_dir'], 'valid', '*.h5')))

    dataset = read_data_train(files=train_files,
                              batch_size=train_params.batch_size,
                              window=3, # For 2-step prediction 
                              num_workers=cfg.get('num_workers', 2),
                              **cfg.get('train_data', {}))
    val_dataset = read_data_val(files=val_files, window=1)

    model = create_model(args.gpu, **cfg.get('model', {}))

    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [10*_k, 20*_k, 30*_k],
        [train_params.base_lr,
         train_params.base_lr*0.5,
         train_params.base_lr*0.25,
         train_params.base_lr*0.1]
    )
    optimizer = tf.keras.optimizers.Adam(lr)

    trainer = Trainer(train_dir)

    ckpt = tf.train.Checkpoint(
        step=tf.Variable(0, dtype=tf.int64),
        model=model,
        optimizer=optimizer
    )

    manager = MyCheckpointManager(
        ckpt,
        trainer.checkpoint_dir,
        keep_checkpoint_steps=list(range(_k, train_params.max_iter+1, _k))
    )

    data_iter = iter(dataset)

    print("[INFO] Start training...")

    if manager.latest_checkpoint:
        print('restoring from ', manager.latest_checkpoint)
        ckpt.restore(manager.latest_checkpoint)

    while trainer.keep_training(
        ckpt.step,
        train_params.max_iter,
        checkpoint_manager=manager
    ):
        batch = next(data_iter)
        batch_tf = {k: [tf.convert_to_tensor(x) for x in v] for k, v in batch.items()}

        total, pos_l, vf_l, zero_l, ent_l, mass_l = train_step(model, optimizer, batch_tf)

        if trainer.current_step % 10 == 0:
            print(
                f"[Step {trainer.current_step}] "
                f"Total={float(total):.4f} | "
                f"Pos={float(pos_l):.4f} | "
                f"VF={float(vf_l):.4f} | "
                f"Zero={float(zero_l):.4f} | "
                f"Entropy={float(ent_l):.4f} | "
                f"Mass={float(mass_l):.4f}"
            )

        if trainer.current_step % (_k) == 0 and val_files:
            print("[INFO] Running evaluation...")
            eval_res = evaluate(
                model, 
                val_dataset,
                frame_skip=cfg.get('evaluation', {}).get('frame_skip', 20),
                **cfg.get('evaluation', {})
            )
            print("[EVAL RESULT]", eval_res)

    model.save_weights(os.path.join(
        train_dir,
        "model_" + date.today().strftime("%Y%m%d") + ".h5"
    ))

    print("[INFO] Training finished.")


if __name__ == "__main__":
    main()