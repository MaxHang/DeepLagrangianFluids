#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_models.py: 对比多个模型在特定场景下的评估指标。

本脚本读取多个由 `evaluate.py` 生成的 JSON 结果文件，并为指定的
场景和指标，生成一张包含所有模型性能曲线的对比图。

这对于可视化消融实验或对比不同超参数模型的效果非常有用。

使用方法:
python compare_models.py --scene_id 1 \
                         --metric position_mse \
                         --labels "Baseline" "Ours" \
                         results/baseline_metrics.json results/ours_metrics.json
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 预定义的美化样式
STYLES = [
    {'color': 'royalblue', 'linestyle': '-'},
    {'color': 'orangered', 'linestyle': '--'},
    {'color': 'seagreen', 'linestyle': '-.'},
    {'color': 'purple', 'linestyle': ':'},
    {'color': 'brown', 'linestyle': '-'},
    {'color': 'pink', 'linestyle': '--'},
]

METRIC_CONFIG = {
    'position_mse': {
        'title': 'Position MSE Comparison',
        'ylabel': 'Position MSE',
        'log_scale': True  # MSE 通常在对数尺度下看得更清楚
    },
    'vf_mse': {
        'title': 'Volume Fraction MSE Comparison',
        'ylabel': 'VF MSE',
        'log_scale': True
    },
    'mass_drift_per_phase': {
        'title': 'Average Mass Drift Comparison',
        'ylabel': 'Relative Mass Drift',
        'log_scale': False,
        'formatter': mticker.PercentFormatter(1.0) # 使用百分比格式
    },
    'kinetic_energy': { # 特殊处理，需要同时画 pred 和 gt
        'title': 'Kinetic Energy Comparison',
        'ylabel': 'Kinetic Energy',
        'log_scale': False
    }
}


def load_metrics_data(file_path: Path) -> Dict[str, Any]:
    """从指定的JSON文件加载评估指标数据。"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[ERROR] Failed to load or parse {file_path}: {e}")
        return None

def plot_comparison(json_files: List[Path], labels: List[str], scene_id: str, metric: str, output_path: Path) -> None:
    """
    为指定的场景和指标，绘制多个模型的对比图。

    Args:
        json_files (List[Path]): 包含多个模型结果的JSON文件路径列表。
        labels (List[str]): 与JSON文件一一对应的模型标签（用于图例）。
        scene_id (str): 要对比的目标场景ID。
        metric (str): 要对比的目标指标键名 (如 'position_mse')。
        output_path (Path): 生成的对比图的保存路径。
    """
    if metric not in METRIC_CONFIG:
        print(f"[ERROR] Invalid metric '{metric}'. Available metrics are: {list(METRIC_CONFIG.keys())}")
        return

    config = METRIC_CONFIG[metric]
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    print(f"[INFO] Generating comparison plot for metric '{metric}' on Scene {scene_id}...")

    for i, file_path in enumerate(json_files):
        model_data = load_metrics_data(file_path)
        if not model_data:
            continue

        if scene_id not in model_data:
            print(f"[WARNING] Scene {scene_id} not found in {file_path}. Skipping.")
            continue
        
        scene_metrics = model_data[scene_id]
        timestamps = scene_metrics.get('timestamps')
        metric_values = scene_metrics.get(metric)
        
        if not timestamps or not metric_values:
            print(f"[WARNING] Metric '{metric}' or timestamps not found for Scene {scene_id} in {file_path}. Skipping.")
            continue

        label = labels[i]
        style = STYLES[i % len(STYLES)]

        # --- 特殊处理动能 ---
        if metric == 'kinetic_energy':
            ke_pred = scene_metrics.get('kinetic_energy_pred')
            ke_gt = scene_metrics.get('kinetic_energy_gt')
            if i == 0 and ke_gt: # 只画一次 GT
                ax.plot(timestamps, ke_gt, label='Ground Truth KE', color='black', linestyle=':', linewidth=2)
            if ke_pred:
                ax.plot(timestamps, ke_pred, label=f'{label} (Pred)', **style)
        
        # --- 处理质量漂移 ---
        elif metric == 'mass_drift_per_phase':
            # 对所有相的漂移取平均值
            avg_drift = np.mean(np.array(metric_values), axis=1)
            ax.plot(timestamps, avg_drift, label=label, **style)

        # --- 处理其他标准指标 ---
        else:
            ax.plot(timestamps, metric_values, label=label, **style)

    ax.set_title(config['title'], fontsize=16)
    ax.set_xlabel('Frame ID', fontsize=12)
    ax.set_ylabel(config['ylabel'], fontsize=12)
    if config.get('log_scale', False):
        ax.set_yscale('log')
    if 'formatter' in config:
        ax.yaxis.set_major_formatter(config['formatter'])

    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[SUCCESS] Comparison plot saved to: {output_path}")

def main() -> None:
    """主执行函数。"""
    parser = argparse.ArgumentParser(
        description="Compare evaluation metrics from multiple simulation results JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("json_files", nargs='+', type=Path, help="Paths to the '_eval_metrics.json' files to compare.")
    parser.add_argument("--labels", nargs='+', required=True, help="A list of labels for each JSON file, used in the plot legend. Must match the number of files.")
    parser.add_argument("--scene_id", "-s", type=str, required=True, help="The ID of the scene to generate the comparison for.")
    parser.add_argument("--metric", "-m", type=str, required=True, choices=list(METRIC_CONFIG.keys()), help="The metric to compare.")
    parser.add_argument("--output", "-o", type=Path, help="Output path for the plot. If not specified, a default name will be generated.")
    
    args = parser.parse_args()

    if len(args.json_files) != len(args.labels):
        print("[ERROR] The number of JSON files must match the number of labels.")
        sys.exit(1)

    # 如果未指定输出路径，则生成一个默认路径
    if args.output is None:
        output_dir = Path("comparison_plots")
        output_dir.mkdir(exist_ok=True)
        args.output = output_dir / f"comparison_scene_{args.scene_id}_{args.metric}.png"

    plot_comparison(args.json_files, args.labels, args.scene_id, args.metric, args.output)

if __name__ == '__main__':
    main()