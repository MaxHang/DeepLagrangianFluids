#!/bin/bash

# ==============================================================================
# 批量生成模型对比图脚本
#
# 功能:
#   - 定义多组对比实验（要对比的指标、场景、模型和标签）。
#   - 循环调用核心绘图脚本 (compare_models.py) 来生成每一张对比图。
#   - 自动为生成的图表创建清晰的文件名。
#
# 使用方法:
#   1. 在下面的 "实验配置" 部分，根据您的需求定义好对比组。
#   2. 确保 compare_models.py 脚本存在且可执行。
#   3. 运行此脚本:
#      ./generate_comparison_plots.sh
# ==============================================================================

# --- 1. 设置脚本的健壮性 ---
set -euo pipefail

# --- 2. 路径与基本配置 ---
SCRIPT_DIR=$(dirname "$0")
COMPARE_SCRIPT_PATH="${SCRIPT_DIR}/compare_models.py"
OUTPUT_DIR="comparison_plots"
mkdir -p "$OUTPUT_DIR"

# --- 3. 实验配置：定义您想要生成的所有对比图 ---

# --- 实验组 1: 消融研究 (Ablation Study) ---
#    对比多个模型变体在同一个场景下的表现
# ABLATION_SCENE_ID="cd_0.17_cf_0.07_density_2000_5000" # 假设我们在场景1上进行消融对比
# ABLATION_SCENE_ID="cd_1.0_cf_0.01_density_800_1500" # 假设我们在场景1上进行消融对比
ABLATION_SCENE_ID="cd_0.29_cf_0.83_density_1000_3000" # 假设我们在场景1上进行消融对比

# 定义消融实验中涉及的模型结果文件和对应的标签
ABLATION_MODELS=(
    "/workspace/xyh_synology/graduate/weights/mix_it_dens_cd_cf_separate_pos_phase_v2/train_mix_v2_mix_v2_20250731/model_weights_2025_08_02.h5_eval_metrics.json"
    "/workspace/xyh_synology/graduate/weights/mix_it_dens_cd_cf_separate_pos_phase_v5/20251021102749/ckpt-93000.index_eval_metrics.json"
    # "results/ablation_mean_agg_metrics.json"
    # "results/ablation_feature_norm_metrics.json"
    # "results/ablation_full_model_metrics.json"
)
ABLATION_LABELS=(
    "Baseline"
    # "DeepSets (Sum)"
    # "Mean Aggregation"
    # "Feature Norm"
    "Ours"
)

# 定义消融实验中要对比的指标
ABLATION_METRICS_TO_PLOT=(
    "position_mse"
    "vf_mse"
    "mass_drift_per_phase"
    "kinetic_energy"
)


# --- 实验组 2: 最终模型泛化能力展示 ---
#    用最终模型，对比它在不同场景下的表现 (注意：compare_models.py 本身不直接支持此功能，
#    但我们可以通过多次调用来模拟。这里我们主要还是对比不同模型。)
#    例如，对比您的最终模型和另一个SOTA模型在峡谷场景的表现
FINAL_MODEL_COMPARISON_SCENE_ID="canyon_scene" # 假设峡谷场景的ID是 'canyon_scene'

FINAL_MODELS_TO_COMPARE=(
    "results/final_model_metrics.json"
    "results/sota_model_metrics.json"
)
FINAL_MODELS_LABELS=(
    "Ours"
    "SOTA Baseline"
)

FINAL_METRICS_TO_PLOT=(
    "position_mse"
    "kinetic_energy"
)


# --- 4. 检查核心脚本是否存在 ---
if [ ! -f "$COMPARE_SCRIPT_PATH" ]; then
    echo "[ERROR] Core comparison script not found at: ${COMPARE_SCRIPT_PATH}"
    exit 1
fi

# --- 5. 执行绘图任务 ---

echo "=========================================================="
echo "          Starting Comparison Plot Generation           "
echo "=========================================================="

# --- 执行实验组 1: 消融研究 ---
echo ""
echo "--- Generating Ablation Study Plots (Scene: ${ABLATION_SCENE_ID}) ---"
for metric in "${ABLATION_METRICS_TO_PLOT[@]}"; do
    
    echo "[INFO] Plotting metric: ${metric}"
    
    output_filename="${OUTPUT_DIR}/ablation_scene_${ABLATION_SCENE_ID}_${metric}.png"
    
    python3 "${COMPARE_SCRIPT_PATH}" \
        "${ABLATION_MODELS[@]}" \
        --scene_id "${ABLATION_SCENE_ID}" \
        --metric "${metric}" \
        --labels "${ABLATION_LABELS[@]}" \
        --output "${output_filename}"

    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Plot saved to ${output_filename}"
    else
        echo "[ERROR] Failed to generate plot for metric: ${metric}"
    fi
done


# # --- 执行实验组 2: 最终模型对比 ---
# echo ""
# echo "--- Generating Final Model Comparison Plots (Scene: ${FINAL_MODEL_COMPARISON_SCENE_ID}) ---"
# for metric in "${FINAL_METRICS_TO_PLOT[@]}"; do

#     echo "[INFO] Plotting metric: ${metric}"

#     output_filename="${OUTPUT_DIR}/final_comp_scene_${FINAL_MODEL_COMPARISON_SCENE_ID}_${metric}.png"

#     python3 "${COMPARE_SCRIPT_PATH}" \
#         --scene_id "${FINAL_MODEL_COMPARISON_SCENE_ID}" \
#         --metric "${metric}" \
#         --labels "${FINAL_MODELS_LABELS[@]}" \
#         "${FINAL_MODELS_TO_COMPARE[@]}" \
#         --output "${output_filename}"

#     if [ $? -eq 0 ]; then
#         echo "[SUCCESS] Plot saved to ${output_filename}"
#     else
#         echo "[ERROR] Failed to generate plot for metric: ${metric}"
#     fi
# done


echo ""
echo "=========================================================="
echo "            All Comparison Plots Generated.               "
echo "=========================================================="
echo "Plots are saved in the '${OUTPUT_DIR}' directory."

# ### 如何使用这个脚本

# 1.  **保存**: 将上述代码保存为 `generate_comparison_plots.sh`，并赋予执行权限 (`chmod +x generate_comparison_plots.sh`)。

# 2.  **配置 (最重要的部分)**:
#     *   打开 `generate_comparison_plots.sh` 文件。
#     *   **在 `实验配置` 部分**，根据您的实际文件路径和需求，修改 `ABLATION_MODELS`, `ABLATION_LABELS`, `FINAL_MODELS_TO_COMPARE` 等数组。
#     *   您可以根据需要，增加更多的**“实验组”**，只需复制粘贴一个 `for` 循环块，并修改相应的配置数组即可。

# 3.  **运行**:
#     ```bash
#     ./generate_comparison_plots.sh
#     ```
#     脚本会依次执行您定义的所有绘图任务，并将生成的 `.png` 文件保存在 `comparison_plots` 目录中。

# ### 这个工作流的优势

# *   **一键复现**: 当您修改了 `compare_models.py` 的绘图样式，或者重新跑了实验更新了 `.json` 文件后，只需运行这一个脚本，就能一键重新生成论文中所有的对比图，保证了图表的一致性和最新状态。
# *   **配置集中管理**: 所有关于“画哪些图”的配置都集中在这个脚本的开头，一目了然，非常便于管理和修改。
# *   **自动化**: 将手动执行多条 `python` 命令的繁琐工作完全自动化，减少了手动输入参数出错的可能性。
# *   **清晰的日志**: 脚本会打印出当前正在生成的图表信息，让您清楚地知道进度。

# 这个“绘图指挥官”脚本，与您之前的“评估指挥官”脚本相配合，构成了一套非常完整和专业的自动化实验流程。