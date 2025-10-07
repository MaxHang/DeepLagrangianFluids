#!/bin/bash

# ==============================================================================
# 批量实验执行脚本 ("指挥官"脚本) - 多GPU并行版
#
# 功能:
#   - 在多块GPU上并行执行多个一一对应的评估实验。
#   - 自动将实验任务分发到指定的GPU上。
#   - 控制并发任务数量，防止GPU过载。
#
# 使用方法:
#   1. 在下面的 "GPU配置" 部分，填入您服务器上可用的GPU ID。
#   2. 在 "实验配置" 部分，配置好模型和配置文件的配对。
#   3. 直接运行:
#      ./run_all_experiments.sh
#      (推荐在 screen 或 tmux 会话中运行，以防连接中断)
# ==============================================================================

# --- 1. 设置脚本的健壮性 ---
set -euo pipefail

# --- 2. GPU 配置 ---
#    在此处定义您希望用于并行评估的GPU ID列表。
#    例如: AVAILABLE_GPUS=(0 1 2 3) for four GPUs
AVAILABLE_GPUS=(0 1 2) 
NUM_GPUS=${#AVAILABLE_GPUS[@]}
echo "[INFO] Starting parallel evaluation on ${NUM_GPUS} GPUs: ${AVAILABLE_GPUS[*]}"

# --- 3. 核心脚本与日志配置 ---
SCRIPT_DIR="scripts"
# 假设您的 train_mix_v2.py 和 evaluate.py 都需要指定 GPU ID
# 我们需要修改 evaluate.py 让它能接收 --gpu 参数
TRAIN_SCRIPT_PATH="${SCRIPT_DIR}/train_mix_v2.py"
EVAL_SCRIPT_PATH="${SCRIPT_DIR}/evaluate_mix_20251008.py"
LOG_DIR="$(dirname "$0")/log_evaluation"
mkdir -p "$LOG_DIR"

# --- 4. 实验配置 (一一对应) ---
MODELS_TO_TEST=(
    /workspace/xyh_synology/graduate/weights/mix_it_dens_cd_cf_separate_pos_phase_v2/phases5_centreTrue_mean_20250811/model_weights_2025_08_14.h5
    /workspace/xyh_synology/graduate/weights/mix_it_dens_cd_cf_separate_pos_phase_v2/phases5_centreTrue_sum_20250811/model_weights_2025_08_14.h5
    /workspace/xyh_synology/graduate/weights/mix_it_dens_cd_cf_separate_pos_phase_v2/train_mix_v2_mix_v2_20250731/model_weights_2025_08_02.h5
    /workspace/xyh_synology/graduate/weights/mix_it_dens_cd_cf_separate_pos_phase_v2/train_mix_v2_mix_v2_20250729/model_weights_2025_08_01.h5
)

CONFIGS_TO_USE=(
    /workspace/xyh_synology/graduate/weights/mix_it_dens_cd_cf_separate_pos_phase_v2/phases5_centreTrue_mean_20250811/training_config.yaml
    /workspace/xyh_synology/graduate/weights/mix_it_dens_cd_cf_separate_pos_phase_v2/phases5_centreTrue_sum_20250811/training_config.yaml
    /workspace/xyh_synology/graduate/weights/mix_it_dens_cd_cf_separate_pos_phase_v2/train_mix_v2_mix_v2_20250731/training_config.yaml
    /workspace/xyh_synology/graduate/weights/mix_it_dens_cd_cf_separate_pos_phase_v2/train_mix_v2_mix_v2_20250729/training_config.yaml
)

# --- 5. 检查配置 ---
if [ ! -f "$TRAIN_SCRIPT_PATH" ]; then
    echo "[ERROR] Training script not found at: ${TRAIN_SCRIPT_PATH}" && exit 1
fi
if [ ! -f "$EVAL_SCRIPT_PATH" ]; then
    echo "[ERROR] Evaluation script not found at: ${EVAL_SCRIPT_PATH}" && exit 1
fi
if [ ${#MODELS_TO_TEST[@]} -ne ${#CONFIGS_TO_USE[@]} ]; then
    echo "[ERROR] Mismatch between number of models and configs!" && exit 1
fi

# --- 6. 定义并行执行函数 ---
#    这个函数负责启动一个单独的评估任务
run_single_evaluation() {
    local model_path="$1"
    local scene_cfg_path="$2"
    local gpu_id="$3"
    local exp_num="$4"
    local total_runs="$5"

    echo "----------------------------------------------------------"
    echo "--> [GPU ${gpu_id}] Starting Experiment ${exp_num} of ${total_runs}"
    echo "----------------------------------------------------------"

    # 检查文件是否存在
    if [ ! -f "$model_path" ] || [ ! -f "$scene_cfg_path" ]; then
        echo "[WARNING][GPU ${gpu_id}] Model or config not found, skipping pair: ${model_path}, ${scene_cfg_path}"
        return
    fi
    
    # 构建日志文件名
    local model_name=$(basename "$model_path" | sed 's/\.index//' | sed 's/\.h5//')
    local scene_name=$(basename "$scene_cfg_path" .yaml)
    local log_filename="${LOG_DIR}/${model_name}_with_${scene_name}_GPU_${gpu_id}.log"
    
    echo "  GPU ID  : ${gpu_id}"
    echo "  Model   : ${model_path}"
    echo "  Config  : ${scene_cfg_path}"
    echo "  Log File: ${log_filename}"
    echo ""
    
    # --- 核心执行命令 ---
    # 使用 CUDA_VISIBLE_DEVICES 来隔离GPU, 其实隔离了之后就不用添加--gpu参数了， 因为export CUDA_VISIBLE_DEVICES="$gpu_id" 为每个后台进程隔离GPU。
    # 注意：我们假设您的 evaluate.py 现在可以接收一个 --gpu 参数
    (
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        python3 "${EVAL_SCRIPT_PATH}" \
            "${TRAIN_SCRIPT_PATH}" \
            --cfg "${scene_cfg_path}" \
            --weights "${model_path}" \
            --gpu "$gpu_id" # 将GPU ID传递给python脚本
    ) > "$log_filename" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "[ERROR][GPU ${gpu_id}] Evaluation failed for pair: ('${model_name}', '${scene_name}'). Check log: ${log_filename}"
    else
        echo "[SUCCESS][GPU ${gpu_id}] Completed experiment for pair: ('${model_name}', '${scene_name}')."
    fi
}


# --- 7. 任务分发与并行控制 ---

echo "=========================================================="
echo "      Starting Parallel Paired Evaluation Experiments      "
echo "=========================================================="

TOTAL_RUNS=${#MODELS_TO_TEST[@]}
job_count=0

# 遍历所有实验配对
for i in "${!MODELS_TO_TEST[@]}"; do
    # 计算当前应该使用哪个GPU (通过取模运算循环使用)
    gpu_index=$((job_count % NUM_GPUS))
    gpu_id="${AVAILABLE_GPUS[$gpu_index]}"
    
    # 将任务放入后台执行
    run_single_evaluation "${MODELS_TO_TEST[$i]}" "${CONFIGS_TO_USE[$i]}" "$gpu_id" "$((i + 1))" "$TOTAL_RUNS" &

    job_count=$((job_count + 1))
    
    # --- 并发控制 ---
    # 如果正在运行的任务数量达到了GPU的数量，就等待其中任何一个完成
    if [ "$job_count" -ge "$NUM_GPUS" ]; then
        wait -n # 等待一个后台任务完成
    fi
done

# --- 8. 等待所有剩余任务完成 ---
echo ""
echo "[INFO] All tasks have been dispatched. Waiting for remaining jobs to finish..."
wait # 等待所有后台任务完成

echo ""
echo "=========================================================="
echo "        All Parallel Evaluation Experiments Finished.        "
echo "=========================================================="
echo "Logs are saved in the '${LOG_DIR}' directory."