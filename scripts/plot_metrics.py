#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
用于可视化多相流体模拟评估指标的脚本。

本脚本读取由 `evaluate.py` 生成的单个 JSON 结果文件，并为其中的
每一个场景生成一组可视化的折线图，保存在与JSON文件相同的目录下。
便于分析模型的长期稳定性和物理一致性。

主要功能:
- 为每个场景自动生成一张包含四个子图的汇总图。
- 可视化关键指标随时间的变化趋势，包括：
  1. 位置均方误差 (Position MSE)
  2. 体积分数均方误差 (VF MSE)
  3. 各相的质量漂移 (Mass Drift per Phase)
  4. 预测与真实的动能 (Kinetic Energy) 对比

使用方法:
python plot_metrics.py path/to/your_eval_metrics.json
"""

import json
import os
import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def load_metrics_data(file_path: Path) -> Dict[str, Any]:
    """
    从指定的JSON文件加载评估指标数据。

    Args:
        file_path (Path): 指向 `_eval_metrics.json` 文件的路径。

    Returns:
        Dict[str, Any]: 包含所有场景评估数据的字典。
    """
    print(f"[INFO] Loading metrics data from: {file_path}")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"[INFO] Successfully loaded data for {len(data)} scene(s).")
        return data
    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"[ERROR] Failed to parse JSON file. It might be corrupted: {file_path}")
        sys.exit(1)

def plot_scene_metrics(scene_id: str, metrics: Dict[str, Any], output_dir: Path) -> None:
    """
    为单个场景的所有指标生成并保存一张汇总图。

    Args:
        scene_id (str): 场景的ID (在JSON中是字符串键)。
        metrics (Dict[str, Any]): 该场景对应的指标数据字典。
        output_dir (Path): 保存生成图表的目录。
    """
    print(f"[INFO] Plotting metrics for Scene {scene_id}...")

    # 创建一个 2x2 的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Evaluation Metrics for Scene {scene_id}', fontsize=16)
    
    # 提取时间戳作为X轴
    timestamps = metrics.get('timestamps', [])
    if not timestamps:
        print(f"[WARNING] No timestamps found for Scene {scene_id}. Skipping plot.")
        plt.close(fig)
        return

    # --- 1. 绘制 Position MSE ---
    ax1 = axes[0, 0]
    pos_mse = metrics.get('position_mse', [])
    if pos_mse:
        ax1.plot(timestamps, pos_mse, label='Position MSE', color='royalblue')
        ax1.set_title('Position MSE over Time')
        ax1.set_xlabel('Frame ID')
        ax1.set_ylabel('MSE')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # 使用科学计数法
    else:
        ax1.text(0.5, 0.5, 'No Position MSE data', ha='center', va='center')
        ax1.set_title('Position MSE over Time')

    # --- 2. 绘制 VF MSE ---
    ax2 = axes[0, 1]
    vf_mse = metrics.get('vf_mse', [])
    if vf_mse:
        ax2.plot(timestamps, vf_mse, label='VF MSE', color='seagreen')
        ax2.set_title('Volume Fraction MSE over Time')
        ax2.set_xlabel('Frame ID')
        ax2.set_ylabel('MSE')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    else:
        ax2.text(0.5, 0.5, 'No VF MSE data', ha='center', va='center')
        ax2.set_title('Volume Fraction MSE over Time')

    # --- 3. 绘制 Mass Drift ---
    ax3 = axes[1, 0]
    mass_drift = np.array(metrics.get('mass_drift_per_phase', []))
    if mass_drift.size > 0:
        num_phases = mass_drift.shape[1]
        for i in range(num_phases):
            ax3.plot(timestamps, mass_drift[:, i], label=f'Phase {i+1} Drift')
        ax3.set_title('Mass Drift per Phase over Time')
        ax3.set_xlabel('Frame ID')
        ax3.set_ylabel('Relative Mass Drift')
        ax3.yaxis.set_major_formatter(mticker.PercentFormatter(1.0)) # 格式化为百分比
        ax3.grid(True, linestyle='--', alpha=0.6)
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No Mass Drift data', ha='center', va='center')
        ax3.set_title('Mass Drift per Phase over Time')

    # --- 4. 绘制 Kinetic Energy ---
    ax4 = axes[1, 1]
    ke_pred = metrics.get('kinetic_energy_pred', [])
    ke_gt = metrics.get('kinetic_energy_gt', [])
    if ke_pred and ke_gt:
        ax4.plot(timestamps, ke_pred, label='Predicted KE', color='orangered')
        ax4.plot(timestamps, ke_gt, label='Ground Truth KE', color='black', linestyle='--')
        ax4.set_title('Kinetic Energy Comparison')
        ax4.set_xlabel('Frame ID')
        ax4.set_ylabel('Kinetic Energy')
        ax4.grid(True, linestyle='--', alpha=0.6)
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No Kinetic Energy data', ha='center', va='center')
        ax4.set_title('Kinetic Energy Comparison')

    # 调整布局并保存图像
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = output_dir / f"scene_{scene_id}_metrics.png"
    plt.savefig(output_path, dpi=150)
    plt.close(fig) # 关闭图形，释放内存
    print(f"[SUCCESS] Plot for Scene {scene_id} saved to {output_path}")

def main() -> None:
    """主执行函数，负责解析参数、加载数据并调用绘图函数。"""
    parser = argparse.ArgumentParser(
        description="Visualize evaluation metrics from a simulation results JSON file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("json_file", type=Path, help="Path to the '_eval_metrics.json' file generated by the evaluation script.")
    # parser.add_argument("--output_dir", "-o", type=Path, default=Path("evaluation_plots"), help="Directory to save the generated plot images.")
    
    args = parser.parse_args()

    # --- [核心修改] ---
    # 将输出目录默认设置为与输入的 JSON 文件相同的目录
    output_dir = Path(args.json_file).parent.joinpath("evaluation_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Plots will be saved in: {output_dir}")
    # ------------------
    
    # 加载数据
    all_metrics_data = load_metrics_data(args.json_file)
    if not all_metrics_data:
        return

    # 为每个场景生成图表
    for scene_id, scene_metrics in all_metrics_data.items():
        plot_scene_metrics(scene_id, scene_metrics, output_dir)
        
    print(f"\n[COMPLETE] All plots have been generated in {output_dir}.")
        
    print("\n[COMPLETE] All plots have been generated.")

if __name__ == '__main__':
    main()