#!/usr/bin/env python3
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

# ===================================================================
# 1. 脚本的依赖项与路径设置
# ===================================================================
# 将父目录添加到系统路径，以便导入项目中的其他模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# 假设您的数据集读取器位于 datasets 目录
from datasets.dataset_reader_h5_mix import read_data_val

# ===================================================================
# 2. 全新的评估指标收集器：SimulationMetrics 类
# ===================================================================
class SimulationMetrics:
    """
    一个用于收集、计算和保存多相流体模拟评估指标的专用类。
    该类按场景(scene)组织数据，并能计算多种准确性和物理一致性指标。
    """
    def __init__(self):
        """初始化一个空的指标字典。"""
        self.metrics_data = {}
        print("[Metrics] SimulationMetrics collector initialized.")

    def _get_or_create_scene(self, scene_id):
        """如果场景ID不存在，则为其创建一个新的条目。"""
        if scene_id not in self.metrics_data:
            self.metrics_data[scene_id] = {
                # 准确性指标 (Accuracy Metrics)
                'position_mse': [],
                'vf_mse': [],
                # 物理一致性指标 (Physical Consistency Metrics)
                'mass_drift_per_phase': [],
                'kinetic_energy_pred': [],
                'kinetic_energy_gt': [],
                # 辅助信息
                'timestamps': []
            }
        return self.metrics_data[scene_id]

    def record_step(self, scene_id, frame_id, pr_pos, gt_pos, pr_vf=None, gt_vf=None, pr_vel=None, gt_vel=None, initial_total_mass=None, particle_masses=None):
        """
        在模拟的单个时间步记录所有相关指标。

        Args:
            scene_id (int): 场景的唯一标识符。
            frame_id (int): 当前的帧ID。
            pr_pos (np.array): 预测的粒子位置。
            gt_pos (np.array): 基准真相的粒子位置。
            pr_vf (np.array, optional): 预测的体积分数。
            gt_vf (np.array, optional): 基准真相的体积分数。
            pr_vel (np.array, optional): 预测的粒子速度。
            gt_vel (np.array, optional): 基准真相的粒子速度。
            initial_total_mass (np.array, optional): 每个相的初始总质量。
            particle_masses (np.array, optional): 每个粒子的质量。
        """
        scene_metrics = self._get_or_create_scene(scene_id)
        
        # --- 记录准确性指标 ---
        # 1. 位置均方误差 (Position MSE)
        pos_mse = np.mean(np.sum((pr_pos - gt_pos)**2, axis=-1))
        scene_metrics['position_mse'].append(pos_mse)

        # 2. 体积分数均方误差 (VF MSE)
        if pr_vf is not None and gt_vf is not None:
            vf_mse = np.mean(np.sum((pr_vf - gt_vf)**2, axis=-1))
            scene_metrics['vf_mse'].append(vf_mse)
        
        # --- 记录物理一致性指标 ---
        # 3. 质量漂移 (Mass Drift)
        if pr_vf is not None and initial_total_mass is not None:
            current_total_mass = np.sum(pr_vf, axis=0)
            # 计算每个相的相对漂移百分比
            mass_drift = np.abs(current_total_mass - initial_total_mass) / initial_total_mass
            scene_metrics['mass_drift_per_phase'].append(mass_drift)

        # 4. (可选) 动能 (Kinetic Energy)
        if pr_vel is not None and gt_vel is not None and particle_masses is not None:
            ke_pred = 0.5 * np.sum(particle_masses * np.sum(pr_vel**2, axis=-1))
            ke_gt = 0.5 * np.sum(particle_masses * np.sum(gt_vel**2, axis=-1))
            scene_metrics['kinetic_energy_pred'].append(ke_pred)
            scene_metrics['kinetic_energy_gt'].append(ke_gt)
        
        scene_metrics['timestamps'].append(frame_id)

    def summarize_results(self):
        """计算并返回所有场景的最终平均指标。"""
        summary = {
            'overall_position_mse': [],
            'overall_vf_mse': [],
            'overall_final_mass_drift': []
        }
        
        for scene_id, metrics in self.metrics_data.items():
            if metrics['position_mse']:
                summary['overall_position_mse'].append(np.mean(metrics['position_mse']))
            if metrics['vf_mse']:
                summary['overall_vf_mse'].append(np.mean(metrics['vf_mse']))
            if metrics['mass_drift_per_phase']:
                # 取最后一个时间步的漂移作为代表，并对所有相取平均
                final_drift_per_phase = metrics['mass_drift_per_phase'][-1]
                summary['overall_final_mass_drift'].append(np.mean(final_drift_per_phase))
        
        # 计算所有场景的最终平均值
        final_results = {
            'Position MSE': np.mean(summary['overall_position_mse']) if summary['overall_position_mse'] else -1,
            'VF MSE': np.mean(summary['overall_vf_mse']) if summary['overall_vf_mse'] else -1,
            'Final Mass Drift': np.mean(summary['overall_final_mass_drift']) if summary['overall_final_mass_drift'] else -1
        }
        return final_results
        
    def save(self, path):
        """将收集到的原始指标数据保存为JSON文件。"""
        serializable_data = {}
        for scene_id, metrics in self.metrics_data.items():
            serializable_data[scene_id] = {k: np.array(v).tolist() for k, v in metrics.items()}
        
        with open(path, 'w') as f:
            json.dump(serializable_data, f, indent=4)
        print(f"[Metrics] Raw metrics data saved to {path}")
            
    def load(self, path):
        """从JSON文件加载指标数据。"""
        with open(path, 'r') as f:
            loaded_data = json.load(f)
        self.metrics_data = loaded_data # JSON库会自动将list转为list，无需转numpy
        print(f"[Metrics] Metrics data loaded from {path}")

# ===================================================================
# 3. 核心评估函数：evaluate_whole_sequence_tf
# ===================================================================
def evaluate_whole_sequence_tf(model, val_dataset, frame_skip, metrics_collector, scale=1, **kwargs):
    """
    执行长时序（rollout）评估，并使用 SimulationMetrics 实例记录所有指标。

    Args:
        model: 训练好的 TensorFlow 模型。
        val_dataset: 验证数据集的迭代器。
        frame_skip (int): 每隔多少帧进行一次评估。
        metrics_collector (SimulationMetrics): 用于记录指标的实例。
        scale (float): 位置缩放因子。
    """
    print('[Evaluation] Starting whole sequence evaluation...')

    last_scene_id = None
    
    # 每个场景的 rollout 状态
    pr_pos, pr_vel, pr_vf = None, None, None
    initial_total_mass_per_phase = None
    particle_masses = None
    box, box_normals = None, None
    scene_phys_params = {}

    for i, data in enumerate(val_dataset):
        scene_id = data['scene_id0'][0]

        # 如果是新场景，进行初始化
        if last_scene_id is None or last_scene_id != scene_id:
            print(f"\n[Evaluation] Processing new scene: {scene_id}...", end='', flush=True)
            last_scene_id = scene_id
            
            # 初始化状态
            pr_pos, pr_vel = data['pos0'][0], data['vel0'][0]
            pr_vf = data.get('phase_fractions0', [None])[0]
            box, box_normals = data['box'][0], data['box_normals'][0]

            # 获取当前场景的物理属性并缓存
            scene_phys_params = {
                'num_phases': tf.constant(data['num_phases'][0], dtype=tf.int32),
                'densities': tf.constant(data['density'][0], dtype=tf.float32),
                'cd': tf.cast(data.get('cd', [0.5])[0], tf.float32),
                'cf': tf.cast(data.get('cf', [0.5])[0], tf.float32)
            }
            
            # 为守恒性计算做准备
            if pr_vf is not None:
                # 初始混合密度
                initial_mixture_density = np.sum(pr_vf * data['density'][0], axis=-1, keepdims=True)
                # 假设所有粒子体积为1，质量即为密度
                particle_masses = initial_mixture_density 
                # 计算每个相的初始总质量 (体积分数*粒子质量)
                initial_total_mass_per_phase = np.sum(pr_vf * particle_masses, axis=0) 
            
        # --- 统一的预测步骤 ---
        inputs = (pr_pos, pr_vel, pr_vf, box, box_normals)
        
        # 使用缓存的物理参数进行预测，成为下一次迭代的输入
        pr_pos_tensor, pr_vel_tensor, pr_vf_tensor = model(
            inputs,
            current_num_phases=scene_phys_params['num_phases'],
            phase_densities=scene_phys_params['densities'],
            cd=scene_phys_params['cd'],
            cf=scene_phys_params['cf'],
            training=False
        )
        pr_pos, pr_vel, pr_vf = pr_pos_tensor.numpy(), pr_vel_tensor.numpy(), pr_vf_tensor.numpy()

        # 在指定的帧上计算并记录误差
        frame_id = data['frame_id0'][0]
        if frame_id > 0 and frame_id % frame_skip == 0:
            print('.', end='', flush=True) # 打印进度点
            gt_pos = data['pos0'][0]
            gt_vf = data.get('phase_fractions0', [None])[0]
            gt_vel = data.get('vel0', [None])[0]

            # 调用 metrics_collector 来记录所有指标
            metrics_collector.record_step(
                scene_id, frame_id,
                pr_pos=scale * pr_pos, gt_pos=scale * gt_pos,
                pr_vf=pr_vf, gt_vf=gt_vf,
                pr_vel=pr_vel, gt_vel=gt_vel,
                initial_total_mass=initial_total_mass_per_phase,
                particle_masses=particle_masses
            )

    print('\n[Evaluation] Whole sequence evaluation finished.')
    return metrics_collector

# ===================================================================
# 4. 主逻辑与辅助函数
# ===================================================================
def print_results(final_results):
    """以格式化的方式打印最终的评估结果。"""
    print('\n==================== FINAL EVALUATION SUMMARY ====================')
    for name, value in final_results.items():
        if 'Drift' in name:
            print(f"  {name:<20}: {value:.4%}")
        else:
            print(f"  {name:<20}: {value:.6f}")
    print('==============================================================')

def eval_checkpoint(checkpoint_path, val_files, options, cfg, train_script_module, gpu_id=0):
    """加载一个检查点并执行评估。"""
    print(f"[Main] Loading validation data...")
    val_dataset = read_data_val(files=val_files, window=1, cache_data=True)
    
    print(f"[Main] Creating model architecture...")
    model = train_script_module.create_model(gpu_id=gpu_id, **cfg.get('model', {}))
    
    print(f"[Main] Restoring model weights from: {checkpoint_path}")
    if checkpoint_path.endswith('.h5'):
        # 对于 .h5 文件，需要先用虚拟数据构建模型
        model.init()
        model.load_weights(checkpoint_path, by_name=True)
    else: # .index for ckpt
        checkpoint = tf.train.Checkpoint(model=model)
        # 移除 .index 后缀以加载检查点
        checkpoint.restore(os.path.splitext(checkpoint_path)[0]).expect_partial()
    print("[Main] Model weights restored successfully.")

    # 创建一个新的 metrics collector 实例
    metrics_collector = SimulationMetrics()
    
    # 执行评估
    metrics_collector = evaluate_whole_sequence_tf(
        model, 
        val_dataset, 
        options.frame_skip, 
        metrics_collector, 
        **cfg.get('evaluation', {})
    )
    
    return metrics_collector

def main():
    parser = argparse.ArgumentParser(
        description="Evaluates a fluid network using a comprehensive set of metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("trainscript", type=str, help="The python training script that defines create_model.")
    parser.add_argument("--cfg", type=str, required=True, help="The path to the yaml config file.")
    parser.add_argument("--weights", type=str, required=True, help="Path to the model weights file (.h5 or .index for ckpt).")
    parser.add_argument("--frame_skip", type=int, default=5, help="The frame skip for evaluation.")
    # --- 新增 ---
    parser.add_argument("--gpu", type=int, default=0, help="The GPU ID to use for this evaluation task.")
    # --- 结束新增 ---
    
    args = parser.parse_args()
    print("[Main] Starting evaluation process with arguments:", args)

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    # 动态导入训练脚本以使用其 create_model 函数
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
        metrics_collector = SimulationMetrics()
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