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
import matplotlib as mpl
from matplotlib import font_manager

# ===================== 字体（关键：强制用路径） =====================
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

if not Path(FONT_PATH).exists():
    raise RuntimeError(f"❌ 中文字体不存在: {FONT_PATH}")

font_prop = font_manager.FontProperties(fname=FONT_PATH)

# ===================== 论文级绘图配置 =====================
def setup_plot_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "axes.unicode_minus": False,

        # 字号（论文推荐）
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,

        # 线条
        "lines.linewidth": 2,

        # 图像
        "figure.figsize": (10, 6),
        "savefig.dpi": 300,
        "savefig.bbox": "tight",

        # 网格
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.4,

        # 去掉上右边框（论文风格）
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# 论文配色
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]
LINESTYLES = ["-", "--", "-.", ":", "-"]

# 指标配置
METRIC_CONFIG = {
    'position_mse': {
        'ylabel': '位置均方误差',
        'log_scale': True
    },
    'vf_mse': {
        'ylabel': '体积分数均方误差',
        'log_scale': True
    },
    'mass_drift_per_phase': {
        'ylabel': '相对质量漂移',
        'log_scale': False,
        'formatter': mticker.PercentFormatter(1.0)
    },
    'kinetic_energy': {
        'ylabel': '动能',
        'log_scale': False
    }
}


def load_metrics_data(file_path: Path) -> Dict[str, Any]:
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
        return None

def plot_all_metrics(json_files, labels, scene_id, output_path):
    setup_plot_style()

    from matplotlib import font_manager
    FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_prop = font_manager.FontProperties(fname=FONT_PATH)

    metrics = [
        "position_mse",
        "vf_mse",
        "mass_drift_per_phase",
        "kinetic_energy"
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    subplot_titles = ["(a)", "(b)", "(c)", "(d)"]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        config = METRIC_CONFIG[metric]

        for i, file_path in enumerate(json_files):
            model_data = load_metrics_data(file_path)
            if not model_data:
                continue

            if scene_id not in model_data:
                continue

            scene_metrics = model_data[scene_id]
            timestamps = scene_metrics.get('timestamps')
            label = labels[i]

            color = COLORS[i % len(COLORS)]
            linestyle = LINESTYLES[i % len(LINESTYLES)]

            if metric == 'kinetic_energy':
                ke_pred = scene_metrics.get('kinetic_energy_pred')
                ke_gt = scene_metrics.get('kinetic_energy_gt')

                if i == 0 and ke_gt:
                    ax.plot(timestamps, ke_gt, color='black', linestyle=':', label='真实值')

                ax.plot(timestamps, ke_pred, label=label, color=color, linestyle=linestyle)

            elif metric == 'mass_drift_per_phase':
                values = np.array(scene_metrics.get(metric))
                avg_drift = np.mean(values, axis=1)
                ax.plot(timestamps, avg_drift, label=label, color=color, linestyle=linestyle)

            else:
                values = scene_metrics.get(metric)
                ax.plot(timestamps, values, label=label, color=color, linestyle=linestyle)

        # 坐标轴
        ax.set_xlabel("时间步", fontproperties=font_prop)
        ax.set_ylabel(config['ylabel'], fontproperties=font_prop)

        if config.get("log_scale"):
            ax.set_yscale("log")

        if "formatter" in config:
            ax.yaxis.set_major_formatter(config["formatter"])

        # 子图标题（放在下面）
        ax.text(
            0.5, -0.28, subplot_titles[idx],
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=13
        )

    # 统一图例（放在顶部）
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_legend,
        loc='upper center',
        ncol=len(labels),
        frameon=False,
        prop=font_prop
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)

    print(f"[SUCCESS] 四合一图已保存: {output_path}")


def plot_comparison(json_files: List[Path], labels: List[str], scene_id: str, metric: str, output_path: Path) -> None:
    setup_plot_style()

    if metric not in METRIC_CONFIG:
        print(f"[ERROR] Invalid metric '{metric}'")
        return

    config = METRIC_CONFIG[metric]

    fig, ax = plt.subplots()
    avg_results = []

    print(f"[INFO] 生成场景 {scene_id} 的 {metric} 对比图...")

    for i, file_path in enumerate(json_files):
        model_data = load_metrics_data(file_path)
        if not model_data:
            continue

        if scene_id not in model_data:
            print(f"[WARNING] 场景 {scene_id} 未在 {file_path} 中找到")
            continue

        scene_metrics = model_data[scene_id]
        timestamps = scene_metrics.get('timestamps')
        label = labels[i]

        color = COLORS[i % len(COLORS)]
        linestyle = LINESTYLES[i % len(LINESTYLES)]

        current_mean = None

        if metric == 'kinetic_energy':
            ke_pred = scene_metrics.get('kinetic_energy_pred')
            ke_gt = scene_metrics.get('kinetic_energy_gt')

            if i == 0 and ke_gt:
                ax.plot(timestamps, ke_gt, label='真实值', color='black', linestyle=':')

            ax.plot(timestamps, ke_pred, label=label, color=color, linestyle=linestyle)
            current_mean = np.mean(ke_pred)

        elif metric == 'mass_drift_per_phase':
            metric_values = np.array(scene_metrics.get(metric))
            avg_drift = np.mean(metric_values, axis=1)

            ax.plot(timestamps, avg_drift, label=label, color=color, linestyle=linestyle)
            current_mean = np.mean(avg_drift)

        else:
            metric_values = scene_metrics.get(metric)

            ax.plot(timestamps, metric_values, label=label, color=color, linestyle=linestyle)
            current_mean = np.mean(metric_values)

        if current_mean is not None:
            avg_results.append((label, current_mean))

    # ===== 打印平均值 =====
    print("\n" + "="*50)
    print(f"📊 场景【{scene_id}】- 指标【{config['ylabel']}】平均值对比")
    print("="*50)
    for lbl, mean_val in avg_results:
        print(f"{lbl}: {mean_val:.6f}")
    print("="*50 + "\n")

    # ===== 图表设置（全部强制字体）=====
    ax.set_xlabel('时间步 (frame)', fontproperties=font_prop)
    ax.set_ylabel(config['ylabel'], fontproperties=font_prop)

    if config.get('log_scale'):
        ax.set_yscale('log')

    if 'formatter' in config:
        ax.yaxis.set_major_formatter(config['formatter'])

    # legend 也必须指定字体
    ax.legend(frameon=False, prop=font_prop)

    plt.savefig(output_path)
    plt.close(fig)

    print(f"[SUCCESS] 图像已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_files", nargs='+', type=Path)
    parser.add_argument("--labels", nargs='+', required=True)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--metric", required=True, choices=list(METRIC_CONFIG.keys()))
    parser.add_argument("--output", type=Path)

    args = parser.parse_args()

    if len(args.json_files) != len(args.labels):
        print("[ERROR] 文件数量与标签数量不一致")
        sys.exit(1)

    if args.output is None:
        output_dir = Path("comparison_plots")
        output_dir.mkdir(exist_ok=True)
        args.output = output_dir / f"{args.scene_id}_{args.metric}.png"

    plot_comparison(args.json_files, args.labels, args.scene_id, args.metric, args.output)
    plot_all_metrics(args.json_files, args.labels, args.scene_id, args.output)


if __name__ == '__main__':
    main()