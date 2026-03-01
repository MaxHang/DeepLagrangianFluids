#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
用于数据驱动多相流体模拟网络的评估脚本（增强诊断版本）。

本脚本的核心功能是执行长时序（rollout）评估，以全面衡量模拟器在
准确性和物理一致性方面的性能。其主要组件包括：

1. SimulationMetrics 类:
   一个专门用于收集、计算和保存评估指标的工具类。它能够按场景记录
   多种指标，如位置均方误差（Position MSE）、体积分数均方误差（VF MSE）
   以及质量漂移（Mass Drift）等，并支持将结果序列化为JSON文件。
   
   [V2 增强] 新增VF守恒性检查、总质量漂移监控、详细的调试输出

2. 评估函数 (evaluate_single_scene_tf):
   负责对单个独立的场景进行完整的自回归（rollout）预测，并在每个
   评估帧上，调用 SimulationMetrics 实例来记录详细的性能指标。

3. 主控流程 (eval_checkpoint, main):
   负责解析命令行参数，加载指定的模型权重和配置文件，并将验证数据集
   按场景进行分组。随后，它会遍历每个场景，调用评估函数，并在每
   个场景评估完成后立即保存当前的累积结果，以确保在长时间评估过程
   中，已完成的工作不会因意外中断而丢失。
"""

import os
import sys
import argparse
import numpy as np
import re
from glob import glob
import time
import importlib
import tensorflow as tf
import yaml
import json
from typing import Dict, List, Any, Optional

# ===================================================================
# 1. 脚本的依赖项与路径设置
# ===================================================================
# 将父目录添加到系统路径，以便导入项目中的其他模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# 假设您的数据集读取器位于 datasets 目录
from datasets.dataset_reader_h5_mix import read_data_val

# ===================================================================
# 2. 增强的评估指标收集器：SimulationMetrics 类
# ===================================================================
class SimulationMetrics:
    """
    一个用于收集、计算和保存多相流体模拟评估指标的专用类。
    该类按场景(scene)组织数据，并能计算多种准确性和物理一致性指标。
    
    [V2 增强]
    - 新增 vf_sum_error: 检查每个粒子的 VF 和是否等于1
    - 新增 total_mass_drift: 所有相的总质量漂移
    - 新增详细的调试输出选项
    """
    def __init__(self, verbose: bool = False) -> None:
        """
        初始化一个空的指标字典。
        
        Args:
            verbose: 是否输出详细的调试信息
        """
        self.metrics_data: Dict[int, Dict[str, List]] = {}
        self.verbose = verbose
        print("[Metrics] SimulationMetrics collector initialized (Enhanced V2).")

    def _get_or_create_scene(self, scene_id: int) -> Dict[str, List]:
        """
        如果场景ID在记录中不存在，则为其创建一个新的条目。

        Args:
            scene_id (int): 场景的唯一标识符。

        Returns:
            Dict[str, List]: 对应场景的指标存储字典。
        """
        if scene_id not in self.metrics_data:
            self.metrics_data[scene_id] = {
                'position_mse': [],
                'vf_mse': [],
                'mass_drift_per_phase': [],
                'kinetic_energy_pred': [],
                'kinetic_energy_gt': [],
                'vf_sum_error': [],           # ✅ 新增：VF守恒误差
                'total_mass_drift': [],        # ✅ 新增：总质量漂移
                'timestamps': []
            }
        return self.metrics_data[scene_id]

    def record_step(self, scene_id: int, frame_id: int, 
                    pr_pos: np.ndarray, gt_pos: np.ndarray, 
                    pr_vf: Optional[np.ndarray] = None, 
                    gt_vf: Optional[np.ndarray] = None, 
                    pr_vel: Optional[np.ndarray] = None, 
                    gt_vel: Optional[np.ndarray] = None, 
                    initial_total_mass: Optional[np.ndarray] = None, 
                    phase_densities: Optional[np.ndarray] = None) -> None:
        """
        在模拟的单个时间步记录所有相关指标（增强版本）。

        Args:
            scene_id (int): 场景的唯一标识符。
            frame_id (int): 当前的帧ID。
            pr_pos (np.ndarray): 预测的粒子位置。
            gt_pos (np.ndarray): 基准真相的粒子位置。
            pr_vf (np.ndarray, optional): 预测的体积分数，shape [N, num_phases]。
            gt_vf (np.ndarray, optional): 基准真相的体积分数。
            pr_vel (np.ndarray, optional): 预测的粒子速度。
            gt_vel (np.ndarray, optional): 基准真相的粒子速度。
            initial_total_mass (np.ndarray, optional): 每个相的初始总质量，shape [num_phases]。
            phase_densities (np.ndarray, optional): 各相的密度，shape [num_phases]。
        """
        scene_metrics = self._get_or_create_scene(scene_id)
        
        # --- 记录准确性指标 ---
        pos_mse = np.mean(np.sum((pr_pos - gt_pos)**2, axis=-1))
        scene_metrics['position_mse'].append(pos_mse)

        if pr_vf is not None and gt_vf is not None:
            vf_mse = np.mean(np.sum((pr_vf - gt_vf)**2, axis=-1))
            scene_metrics['vf_mse'].append(vf_mse)
        
        # --- 记录物理一致性指标（增强版）---
        if pr_vf is not None and initial_total_mass is not None and phase_densities is not None:
            # ✅ 新增：检查VF是否守恒（Σ VF_i = 1）
            vf_sum = np.sum(pr_vf, axis=-1)  # [N]
            vf_conservation_error = np.mean(np.abs(vf_sum - 1.0))
            scene_metrics['vf_sum_error'].append(vf_conservation_error)
            
            # 质量漂移计算（按相）
            current_mixture_density = np.sum(pr_vf * phase_densities, axis=-1, keepdims=True)  # [N, 1]
            current_total_mass = np.sum(pr_vf * current_mixture_density, axis=0)  # [num_phases]
            mass_drift = (current_total_mass - initial_total_mass) / (initial_total_mass + 1e-10)
            scene_metrics['mass_drift_per_phase'].append(mass_drift)
            
            # ✅ 新增：总质量漂移（所有相之和）
            total_mass_all_phases = np.sum(current_total_mass)
            initial_total_mass_all = np.sum(initial_total_mass)
            total_mass_drift = np.abs(total_mass_all_phases - initial_total_mass_all) / (initial_total_mass_all + 1e-10)
            scene_metrics['total_mass_drift'].append(total_mass_drift)
            
            # ✅ 调试输出（每100帧或检测到异常时）
            if self.verbose and (frame_id % 100 == 0 or vf_conservation_error > 0.01 or np.any(mass_drift > 0.5)):
                print(f"\n[Scene {scene_id}, Frame {frame_id}] Physics Diagnostics:")
                print(f"  VF sum error: {vf_conservation_error:.6f} (should be ~0)")
                print(f"  VF sum range: [{np.min(vf_sum):.6f}, {np.max(vf_sum):.6f}] (should be ~1.0)")
                print(f"  Mass drift per phase (signed):")
                for i, drift in enumerate(mass_drift):
                    sign = "+" if drift > 0 else ""
                    print(f"    Phase {i+1}: {sign}{drift:.4%}")
                print(f"  Total mass drift: {total_mass_drift:.4%}")
                if np.any(mass_drift > 1.0):
                    print(f"  WARNING  WARNING: Mass drift >100% detected!")
                    print(f"  Current mass: {current_total_mass}")
                    print(f"  Initial mass: {initial_total_mass}")

        if pr_vel is not None and gt_vel is not None and pr_vf is not None and phase_densities is not None:
            # 动能计算
            current_mixture_density = np.sum(pr_vf * phase_densities, axis=-1, keepdims=True)  # [N, 1]
            ke_pred = 0.5 * np.sum(current_mixture_density[:, 0] * np.sum(pr_vel**2, axis=-1))
            ke_gt = 0.5 * np.sum(current_mixture_density[:, 0] * np.sum(gt_vel**2, axis=-1))
            scene_metrics['kinetic_energy_pred'].append(ke_pred)
            scene_metrics['kinetic_energy_gt'].append(ke_gt)
        
        scene_metrics['timestamps'].append(frame_id)

    def summarize_results(self) -> Dict[str, Any]:
        """
        计算并返回所有已记录场景的最终平均指标（增强版本）。

        Returns:
            Dict[str, Any]: 包含最终平均指标的字典。
        """
        summary = { 
            'overall_position_mse': [], 
            'overall_vf_mse': [], 
            'overall_final_mass_drift': [],
            'overall_final_mass_drift_per_phase': [],
            'overall_vf_sum_error': [],           # ✅ 新增
            'overall_total_mass_drift': []        # ✅ 新增
        }
        
        for scene_id, metrics in self.metrics_data.items():
            if metrics['position_mse']:
                summary['overall_position_mse'].append(np.mean(metrics['position_mse']))
            if metrics['vf_mse']:
                summary['overall_vf_mse'].append(np.mean(metrics['vf_mse']))
            if metrics['mass_drift_per_phase']:
                final_drift_per_phase = metrics['mass_drift_per_phase'][-1]
                summary['overall_final_mass_drift'].append(np.mean(np.abs(final_drift_per_phase)))
                summary['overall_final_mass_drift_per_phase'].append(final_drift_per_phase)
            if metrics['vf_sum_error']:
                summary['overall_vf_sum_error'].append(np.mean(metrics['vf_sum_error']))
            if metrics['total_mass_drift']:
                summary['overall_total_mass_drift'].append(np.mean(metrics['total_mass_drift']))
        
        # 计算所有场景的最终平均值
        final_results = {
            'Position MSE': np.mean(summary['overall_position_mse']) if summary['overall_position_mse'] else -1.0,
            'VF MSE': np.mean(summary['overall_vf_mse']) if summary['overall_vf_mse'] else -1.0,
            'Final Mass Drift (Average)': np.mean(summary['overall_final_mass_drift']) if summary['overall_final_mass_drift'] else -1.0,
            'Final Mass Drift (Per Phase)': np.mean(summary['overall_final_mass_drift_per_phase'], axis=0).tolist() if summary['overall_final_mass_drift_per_phase'] else [],
            'VF Sum Error (Average)': np.mean(summary['overall_vf_sum_error']) if summary['overall_vf_sum_error'] else -1.0,  # ✅ 新增
            'Total Mass Drift (Average)': np.mean(summary['overall_total_mass_drift']) if summary['overall_total_mass_drift'] else -1.0  # ✅ 新增
        }
        return final_results
        
    def save(self, path: str) -> None:
        """将收集到的原始指标数据安全地保存为JSON文件。"""
        serializable_data = {}
        for scene_id, metrics in self.metrics_data.items():
            serializable_data[scene_id] = {k: np.array(v).tolist() for k, v in metrics.items()}
        
        with open(path, 'w') as f:
            json.dump(serializable_data, f, indent=4)
        print(f"\n[Metrics] Progress saved to {path}")
            
    def load(self, path: str) -> None:
        """从JSON文件加载指标数据。"""
        with open(path, 'r') as f:
            loaded_data = json.load(f)
        self.metrics_data = loaded_data
        print(f"[Metrics] Metrics data loaded from {path}")

# ===================================================================
# 3. 核心评估函数：evaluate_single_scene_tf
# ===================================================================
def evaluate_single_scene_tf(model: tf.keras.Model, scene_data: List[Dict], scene_id: int, frame_skip: int, metrics_collector: SimulationMetrics, scale: float = 1.0) -> SimulationMetrics:
    """
    对单个场景执行完整的长时序（rollout）评估。

    Args:
        model (tf.keras.Model): 训练好的 TensorFlow 模型。
        scene_data (List[Dict]): 属于同一个场景的所有帧的数据列表。
        scene_id (int): 当前场景的ID。
        frame_skip (int): 每隔多少帧进行一次评估。
        metrics_collector (SimulationMetrics): 用于记录指标的实例。
        scale (float): 位置缩放因子。

    Returns:
        SimulationMetrics: 更新后的指标收集器实例。
    """
    print(f"\n[Evaluation] Processing scene: {scene_id}...", end='', flush=True)
    
    # --- 1. 初始化场景状态 ---
    initial_data = scene_data[0]
    pr_pos, pr_vel = initial_data['pos0'][0], initial_data['vel0'][0]
    pr_vf = initial_data.get('phase_fractions0', [None])[0]
    box, box_normals = initial_data['box'][0], initial_data['box_normals'][0]

    # 缓存物理参数
    scene_phys_params = {
        'num_phases': tf.constant(initial_data['num_phases'][0], dtype=tf.int32),
        'densities': tf.constant(initial_data['density'][0], dtype=tf.float32),
        'cd': tf.cast(initial_data.get('cd', [0.5])[0], tf.float32),
        'cf': tf.cast(initial_data.get('cf', [0.5])[0], tf.float32)
    }

    # 为守恒性计算做准备
    phase_densities = initial_data['density'][0]  # [num_phases]
    initial_mixture_density = np.sum(pr_vf * phase_densities, axis=-1, keepdims=True)  # [N, 1]
    initial_total_mass_per_phase = np.sum(pr_vf * initial_mixture_density, axis=0)  # [num_phases]
    
    # ✅ 新增：初始状态验证
    if metrics_collector.verbose:
        initial_vf_sum = np.sum(pr_vf, axis=-1)
        print(f"\n[Scene {scene_id}] Initial State Check:")
        print(f"  Num particles: {pr_vf.shape[0]}")
        print(f"  Num phases: {pr_vf.shape[1]}")
        print(f"  Initial VF sum range: [{np.min(initial_vf_sum):.6f}, {np.max(initial_vf_sum):.6f}]")
        print(f"  Initial mass per phase: {initial_total_mass_per_phase}")
        print(f"  Phase densities: {phase_densities}")

    # --- 2. 循环执行 Rollout ---
    for data in scene_data:
        inputs = (pr_pos, pr_vel, pr_vf, box, box_normals)
        
        pr_pos_tensor, pr_vel_tensor, pr_vf_tensor = model(
            inputs,
            current_num_phases=scene_phys_params['num_phases'],
            phase_densities=scene_phys_params['densities'],
            cd=scene_phys_params['cd'],
            cf=scene_phys_params['cf'],
            training=False
        )
        pr_pos, pr_vel, pr_vf = pr_pos_tensor.numpy(), pr_vel_tensor.numpy(), pr_vf_tensor.numpy()

        frame_id = data['frame_id0'][0]
        if frame_id > 0 and frame_id % frame_skip == 0:
            print('.', end='', flush=True) # 打印进度点
            
            # 记录指标
            metrics_collector.record_step(
                scene_id=scene_id, frame_id=frame_id,
                pr_pos=scale * pr_pos, gt_pos=scale * data['pos0'][0],
                pr_vf=pr_vf, gt_vf=data.get('phase_fractions0', [None])[0],
                pr_vel=pr_vel, gt_vel=data.get('vel0', [None])[0],
                initial_total_mass=initial_total_mass_per_phase,
                phase_densities=phase_densities
            )
            
    print(f" scene finished.", flush=True)
    return metrics_collector

# ===================================================================
# 4. 主控流程与辅助函数
# ===================================================================
def print_results(final_results: Dict[str, Any]) -> None:
    """以格式化的方式打印最终的评估结果（增强版本）。"""
    print('\n==================== FINAL EVALUATION SUMMARY ====================')
    for name, value in final_results.items():
        if name == 'Final Mass Drift (Per Phase)':
            # 特殊处理：按相打印
            print(f"  {name:<35}:")
            if isinstance(value, list) and len(value) > 0:
                for i, drift in enumerate(value):
                    # ✅ 修改：显示正负号和状态
                    abs_drift = abs(drift)
                    sign = "+" if drift > 0 else ("-" if drift < 0 else " ")
                    status = "True" if abs_drift < 0.01 else ("WARNING" if abs_drift < 0.1 else "FALSE")
                    print(f"    Phase {i+1}: {sign}{abs_drift:.4%} {status}")
            else:
                print(f"    N/A")
        elif 'Drift' in name or 'Error' in name:
            status = "True" if value < 0.01 else ("WARNING" if value < 0.1 else "FALSE")
            print(f"  {name:<35}: {value:.4%} {status}")
        else:
            print(f"  {name:<35}: {value:.6f}")
    print('==================================================================')

def eval_checkpoint(checkpoint_path: str, val_files: List[str], options: argparse.Namespace, cfg: Dict, train_script_module: Any, gpu_id: int) -> SimulationMetrics:
    """加载一个检查点，按场景执行评估，并在每个场景后保存。"""
    print(f"[Main] Loading and grouping validation data by scene...")
    val_dataset = read_data_val(files=val_files, window=1, cache_data=True)
    
    # 按场景ID对数据进行分组
    scenes: Dict[int, List] = {}
    for data in val_dataset:
        scene_id = data['scene_id0'][0]
        if scene_id not in scenes:
            scenes[scene_id] = []
        scenes[scene_id].append(data)
    print(f"[Main] Found {len(scenes)} unique scenes in the validation set.")

    print(f"[Main] Creating model architecture on GPU {gpu_id}...")
    model = train_script_module.create_model(gpu_id=gpu_id, **cfg.get('model', {}))
    
    print(f"[Main] Restoring model weights from: {checkpoint_path}")
    if checkpoint_path.endswith('.h5'):
        model.init()
        model.load_weights(checkpoint_path, by_name=True)
    else: # .index for ckpt
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(os.path.splitext(checkpoint_path)[0]).expect_partial()
    print("[Main] Model weights restored successfully.")

    # 定义输出文件路径
    output_filename = os.path.basename(checkpoint_path) + '_eval_metrics.json'
    output_path = os.path.join(os.path.dirname(checkpoint_path), output_filename)
    
    # ✅ 启用详细模式（根据命令行参数）
    metrics_collector = SimulationMetrics(verbose=options.verbose)
    
    # 检查是否可以从已有的结果文件中恢复进度
    if os.path.isfile(output_path):
        print(f"[Main] Found existing results file. Loading progress from: {output_path}")
        metrics_collector.load(output_path)

    # 循环遍历每个场景进行评估
    for scene_id, scene_data in sorted(scenes.items()):
        # 如果该场景的结果已经存在，则跳过
        if scene_id in metrics_collector.metrics_data:
            print(f"\n[Evaluation] Scene {scene_id} already evaluated. Skipping.")
            continue
            
        # 对单个场景执行评估
        metrics_collector = evaluate_single_scene_tf(
            model, 
            scene_data,
            scene_id,
            options.frame_skip, 
            metrics_collector
        )
        # 每个场景评估完毕后，立即保存进度
        metrics_collector.save(output_path)
    
    return metrics_collector

def main() -> int:
    """主执行函数。"""
    parser = argparse.ArgumentParser(description="Evaluates a fluid network with comprehensive metrics and per-scene saving.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("trainscript", type=str, help="The python training script that defines create_model.")
    parser.add_argument("--cfg", type=str, required=True, help="The path to the yaml config file.")
    parser.add_argument("--weights", type=str, required=True, help="Path to the model weights file (.h5 or .index for ckpt).")
    parser.add_argument("--gpu", type=int, default=0, help="The GPU ID to use for this evaluation task.")
    parser.add_argument("--frame_skip", type=int, default=5, help="The frame skip for evaluation.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debugging output.")  # ✅ 新增
    
    args = parser.parse_args()
    print("[Main] Starting evaluation process with arguments:", args)

    with open(args.cfg, 'r') as f:
        cfg = yaml.unsafe_load(f)

    # 动态导入训练脚本
    module_name = os.path.splitext(os.path.basename(args.trainscript))[0]
    sys.path.append('.')
    train_script_module = importlib.import_module(module_name)

    # 查找验证文件
    val_files = sorted(glob(os.path.join(cfg['dataset_dir'], 'valid', '*.h5')))
    if not val_files:
        sys.exit(f"[Error] No validation files found in {os.path.join(cfg['dataset_dir'], 'valid')}")

    # 定义输出文件路径
    output_filename = os.path.basename(args.weights) + '_eval_metrics.json'
    output_path = os.path.join(os.path.dirname(args.weights), output_filename)

    if os.path.isfile(output_path):
        print(f"[Main] Evaluation file already exists: {output_path}")
        print("[Main] Loading previously computed results...")
        metrics_collector = SimulationMetrics(verbose=args.verbose)
        metrics_collector.load(output_path)
    else:
        print(f"[Main] Evaluating weights file: {args.weights}")
        metrics_collector = eval_checkpoint(args.weights, val_files, args, cfg, train_script_module, gpu_id=args.gpu)
        # 保存原始数据以供未来分析
        metrics_collector.save(output_path)

    # 计算并打印最终的汇总结果
    final_results = metrics_collector.summarize_results()
    print_results(final_results)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())