#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VF Prediction Debug Script

Usage:
    python scripts/debug_vf_prediction.py <config.yaml> [--checkpoint <ckpt_dir>]

Diagnostics:
  1. Training data VF change magnitude distribution
  2. Model inference delta_vf magnitude (check if alpha is appropriate)
  3. cd/cf sensitivity (same state, different cd/cf, VF output difference)
  4. VF loss gradient magnitudes for vf_per_phase_mlp and cd_cf_encoder
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import tensorflow as tf
import yaml
import argparse
from glob import glob

from models.default_tf_mix_separate_pos_phase_v6 import MultiPhaseParticleNetwork
from datasets.dataset_reader_h5_mix import read_data_val


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data VF change magnitude distribution
# ─────────────────────────────────────────────────────────────────────────────

def check_data_vf_change(val_files, max_samples=500):
    print("\n" + "="*60)
    print("[Diagnostics 1] VF frame-by-frame change in dataset")
    print("="*60)

    from datasets.dataset_reader_h5_mix import read_data_val
    dataset = read_data_val(files=val_files, window=2, cache_data=False)

    deltas = []
    for i, batch in enumerate(dataset):
        if i >= max_samples:
            break
        vf0 = np.array(batch['phase_fractions0'])
        vf1 = np.array(batch['phase_fractions1'])
        delta = np.abs(vf1 - vf0)
        deltas.append(delta)

    if not deltas:
        print("  Cannot read data, check file paths")
        return

    deltas = np.concatenate(deltas, axis=0)
    print(f"  Samples:            {deltas.shape[0]} particles")
    print(f"  mean |delta_vf|:   {deltas.mean():.6f}")
    print(f"  median |delta_vf|: {np.median(deltas):.6f}")
    print(f"  p95 |delta_vf|:    {np.percentile(deltas, 95):.6f}")
    print(f"  max  |delta_vf|:   {deltas.max():.6f}")
    print(f"  ratio |delta_vf|>0.1: {(deltas > 0.1).mean()*100:.1f}%")
    print(f"  ratio |delta_vf|>0.01: {(deltas > 0.01).mean()*100:.1f}%")
    print(f"\n  If p95 > alpha(0.1), alpha is too small, VF updates are clipped")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model inference delta_vf magnitude
# ─────────────────────────────────────────────────────────────────────────────

class ModelWithDeltaVF(tf.keras.Model):
    """Wrapper to expose delta_vf as extra output"""
    def __init__(self, model):
        super().__init__()
        self._model = model

    def call(self, inputs, phase_densities=None, training=False, **kwargs):
        pos1, vel1, current_phase_fractions, box_pos, box_feats = inputs
        num_phases = tf.shape(current_phase_fractions)[1]
        if phase_densities is None:
            phase_densities = tf.ones([num_phases], dtype=tf.float32) * 1000.0
        phase_densities = tf.convert_to_tensor(phase_densities, dtype=tf.float32)

        pos2, vel2 = self._model.integrate_pos_vel(pos1, vel1)
        per_phase_features, phase_embedding = self._model._encode_phases(
            current_phase_fractions, phase_densities, num_phases)
        fluid_feats = self._model._build_fluid_feats(pos2, vel2, phase_embedding, **kwargs)
        shared_features = self._model._backbone(fluid_feats, pos2, box_pos, box_feats)

        filter_extent = tf.constant(self._model.filter_extent)
        pos_correction = (1.0 / 128.0) * (
            self._model.pos_conv(shared_features, pos2, pos2, filter_extent)
            + self._model.pos_dense(shared_features)
        )
        pos_final, vel_final = self._model.compute_new_pos_vel(pos1, vel1, pos2, vel2, pos_correction)

        vf_spatial = self._model.vf_context_conv(shared_features, pos_final, pos_final, filter_extent)
        vf_spatial_expanded = tf.tile(tf.expand_dims(vf_spatial, axis=1), [1, num_phases, 1])
        per_phase_input = tf.concat([vf_spatial_expanded, per_phase_features], axis=-1)

        delta_vf_raw = tf.squeeze(self._model.vf_per_phase_mlp(per_phase_input), axis=-1)
        delta_vf = tf.keras.activations.tanh(delta_vf_raw)

        vf_next = tf.keras.activations.relu(current_phase_fractions + self._model.alpha * delta_vf)
        s = tf.maximum(tf.reduce_sum(vf_next, axis=-1, keepdims=True), 1e-8)
        vf_out = vf_next / s

        return pos_final, vel_final, vf_out, delta_vf_raw, delta_vf


def check_delta_vf_stats(model, val_files, max_samples=100):
    print("\n" + "="*60)
    print("[Diagnostics 2] Inference delta_vf magnitude (check if network predicts changes)")
    print("="*60)

    wrapped = ModelWithDeltaVF(model)
    dataset = read_data_val(files=val_files, window=1, cache_data=False)

    raw_vals, tanh_vals, vf_changes = [], [], []

    for i, batch in enumerate(dataset):
        if i >= max_samples:
            break
        pos0   = tf.constant(batch['pos0'][0], dtype=tf.float32)
        vel0   = tf.constant(batch['vel0'][0], dtype=tf.float32)
        vf0    = tf.constant(batch['phase_fractions0'][0], dtype=tf.float32)
        box    = tf.constant(batch['box'][0], dtype=tf.float32)
        boxn   = tf.constant(batch['box_normals'][0], dtype=tf.float32)
        dens   = tf.constant(batch['density'][0], dtype=tf.float32)
        cd_v   = float(batch['cd'][0])
        cf_v   = float(batch['cf'][0])

        _, _, vf_out, dv_raw, dv_tanh = wrapped(
            (pos0, vel0, vf0, box, boxn),
            phase_densities=dens,
            cd=np.float32(cd_v), cf=np.float32(cf_v),
        )
        raw_vals.append(dv_raw.numpy())
        tanh_vals.append(dv_tanh.numpy())
        vf_changes.append(np.abs(vf_out.numpy() - vf0.numpy()))

    raw_all  = np.concatenate(raw_vals)
    tanh_all = np.concatenate(tanh_vals)
    vfc_all  = np.concatenate(vf_changes)

    print(f"  delta_vf (raw MLP output):")
    print(f"    mean={raw_all.mean():.4f}  std={raw_all.std():.4f}  "
          f"min={raw_all.min():.4f}  max={raw_all.max():.4f}")
    print(f"  delta_vf (after tanh, alpha={model.alpha}):")
    print(f"    mean={tanh_all.mean():.4f}  std={tanh_all.std():.4f}  "
          f"max_abs={np.abs(tanh_all).max():.4f}")
    print(f"  Actual VF change |vf_out - vf_in|:")
    print(f"    mean={vfc_all.mean():.6f}  max={vfc_all.max():.6f}")
    print()
    if np.abs(tanh_all).max() < 0.01:
        print("  delta_vf near 0 -> MLP collapsed, no VF prediction")
    elif np.abs(raw_all).max() < 0.1:
        print("  MLP output range too small -> vanishing gradient or low VF loss weight")
    else:
        print("  MLP has non-zero output, check alpha matches data scale")


# ─────────────────────────────────────────────────────────────────────────────
# 3. cd/cf sensitivity test
# ─────────────────────────────────────────────────────────────────────────────

def check_cdcf_sensitivity(model, val_files, max_samples=50):
    print("\n" + "="*60)
    print("[Diagnostics 3] cd/cf sensitivity: same input, different cd/cf -> VF difference")
    print("="*60)

    dataset = read_data_val(files=val_files, window=1, cache_data=False)

    diffs = []
    for i, batch in enumerate(dataset):
        if i >= max_samples:
            break
        pos0 = tf.constant(batch['pos0'][0], dtype=tf.float32)
        vel0 = tf.constant(batch['vel0'][0], dtype=tf.float32)
        vf0  = tf.constant(batch['phase_fractions0'][0], dtype=tf.float32)
        box  = tf.constant(batch['box'][0], dtype=tf.float32)
        boxn = tf.constant(batch['box_normals'][0], dtype=tf.float32)
        dens = tf.constant(batch['density'][0], dtype=tf.float32)

        inputs = (pos0, vel0, vf0, box, boxn)

        _, _, vf_no_mix = model(inputs, phase_densities=dens,
                                cd=np.float32(0.0), cf=np.float32(0.0))
        _, _, vf_full_mix = model(inputs, phase_densities=dens,
                                  cd=np.float32(1.0), cf=np.float32(0.0))
        diffs.append(np.abs(vf_full_mix.numpy() - vf_no_mix.numpy()))

    diffs_all = np.concatenate(diffs)
    print(f"  |VF(cd=1) - VF(cd=0)|:")
    print(f"    mean={diffs_all.mean():.6f}  max={diffs_all.max():.6f}  "
          f"p95={np.percentile(diffs_all, 95):.6f}")
    print()
    if diffs_all.mean() < 1e-4:
        print("  VF does NOT respond to cd -> cd/cf signal not reaching VF head")
        print("    Cause: cd/cf gradients overwhelmed by position loss")
    else:
        print("  VF responds to cd, mean difference magnitude:", diffs_all.mean())


# ─────────────────────────────────────────────────────────────────────────────
# 4. Gradient magnitude check
# ─────────────────────────────────────────────────────────────────────────────

def check_gradient_magnitudes(model, val_files, max_samples=10):
    print("\n" + "="*60)
    print("[Diagnostics 4] VF loss gradient magnitudes for subnetworks")
    print("="*60)

    dataset = read_data_val(files=val_files, window=2, cache_data=False)

    for i, batch in enumerate(dataset):
        if i >= max_samples:
            break
        pos0 = tf.constant(batch['pos0'][0], dtype=tf.float32)
        vel0 = tf.constant(batch['vel0'][0], dtype=tf.float32)
        vf0  = tf.constant(batch['phase_fractions0'][0], dtype=tf.float32)
        vf1  = tf.constant(batch['phase_fractions1'][0], dtype=tf.float32)
        box  = tf.constant(batch['box'][0], dtype=tf.float32)
        boxn = tf.constant(batch['box_normals'][0], dtype=tf.float32)
        dens = tf.constant(batch['density'][0], dtype=tf.float32)
        cd_v = float(batch['cd'][0])
        cf_v = float(batch['cf'][0])

        with tf.GradientTape() as tape:
            _, _, vf_out = model(
                (pos0, vel0, vf0, box, boxn),
                phase_densities=dens,
                cd=np.float32(cd_v), cf=np.float32(cf_v),
                training=True,
            )
            vf_loss = tf.reduce_mean(tf.square(vf_out - vf1))

        target_vars = {
            'vf_per_phase_mlp': model.vf_per_phase_mlp.trainable_variables,
            'vf_context_conv':  model.vf_context_conv.trainable_variables,
            'cd_cf_encoder':    model.cd_cf_encoder.trainable_variables if model.cd_cf_as_input else [],
            'phase_encoder':    model.phase_encoder.trainable_variables,
        }
        flat_vars = []
        for name, vs in target_vars.items():
            flat_vars.extend(vs)

        grads = tape.gradient(vf_loss, flat_vars)

        idx = 0
        for name, vs in target_vars.items():
            n = len(vs)
            g_slice = grads[idx:idx+n]
            idx += n
            if not g_slice:
                print(f"  {name:<20}: (no params)")
                continue
            g_norms = [tf.reduce_mean(tf.abs(g)).numpy() if g is not None else 0.0
                       for g in g_slice]
            print(f"  {name:<20}: mean |grad|={np.mean(g_norms):.2e}  "
                  f"max |grad|={np.max(g_norms):.2e}")

        break

    print()
    print("  Reference: normal gradient magnitude 1e-4 ~ 1e-2")
    print("  If vf_per_phase_mlp grad <1e-5: low VF loss weight or vanishing gradient")
    print("  If cd_cf_encoder grad <1e-6: no effective gradient path for cd/cf")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str, help='yaml config path')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint directory')
    args = parser.parse_args()

    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    val_files = sorted(glob(os.path.join(cfg['dataset_dir'], 'valid', '*.h5')))
    if not val_files:
        print(f"Error: no validation files found: {cfg['dataset_dir']}/valid/")
        sys.exit(1)
    print(f"Validation files: {len(val_files)}")

    check_data_vf_change(val_files)

    model = MultiPhaseParticleNetwork(**cfg.get('model', {}))
    model.init()

    if args.checkpoint:
        if args.checkpoint.endswith('.h5'):
            model.load_weights(args.checkpoint, by_name=True)
            print(f"\load h5 weights: {args.checkpoint}")
        else:
            ckpt = tf.train.Checkpoint(model=model)
            latest = tf.train.latest_checkpoint(args.checkpoint)
            if latest:
                ckpt.restore(latest).expect_partial()
                print(f"\nload checkpoint: {latest}")
            else:
                print(f"\ncheckpoint not have weight: {args.checkpoint}")
        ckpt = tf.train.Checkpoint(model=model)
        status = ckpt.restore(tf.train.latest_checkpoint(args.checkpoint))
        print(f"\nLoaded checkpoint: {tf.train.latest_checkpoint(args.checkpoint)}")
    else:
        train_dir = cfg.get('train_dir', '')
        ckpt_dir = os.path.join(train_dir, 'checkpoints')
        if os.path.exists(ckpt_dir) and tf.train.latest_checkpoint(ckpt_dir):
            ckpt = tf.train.Checkpoint(model=model)
            ckpt.restore(tf.train.latest_checkpoint(ckpt_dir)).expect_partial()
            print(f"\nLoaded checkpoint: {tf.train.latest_checkpoint(ckpt_dir)}")
        else:
            print("\n No checkpoint found, using random init (diagnostics 2/3/4 may be meaningless)")

    check_delta_vf_stats(model, val_files)
    check_cdcf_sensitivity(model, val_files)
    check_gradient_magnitudes(model, val_files)

    print("\n" + "="*60)
    print("Diagnostics complete. Fix suggestions below:")
    print("="*60)
    print("""
Fix Suggestions:

[1] p95 |delta_vf| > alpha
  -> Increase alpha (0.3~0.5), or predict vf_next directly (remove residual clamp)

[2] MLP output near 0
  -> Increase loss_weights.vol (1.0 -> 5.0~10.0)
  -> Check if tf.debugging.enable_check_numerics() clamps gradients

[3] VF not responding to cd
  -> Concatenate cd_cf_emb directly in _predict_next_vf for direct signal path
  -> Use higher LR for cd_cf_encoder

[4] vf_per_phase_mlp gradient too small
  -> Increase loss_weights.vol
  -> Add loss scaling for VF loss
""")


if __name__ == '__main__':
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    main()