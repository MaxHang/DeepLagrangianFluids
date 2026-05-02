#!/bin/bash

# 定义变量, 批量处理文件夹
RUN_SCRIPT="scripts/run_mix_v7.py"
TRAIN_SCRIPT="scripts/train_mix_v7.py"
DATE=$(date +"%Y%m%d_%H%M%S")  # 获取当前日期和时间，格式为 YYYYMMDD_HHMMSS
# NUMSTEPS=2000
NUMSTEPS=2000
GPU_ID="0"  # 默认使用GPU 0


LOG_FILE="scripts_mix_it_dens_cd_cf/log_run/run_v7_dataset_20260111_box_3phase_scene_${DATE}.log"  # 定义日志文件名，包含日期和时间
OUTPUT="/workspace/xyh_synology/graduate/run_network/mix_fluid/ply_dataset_20260111_box_3phase_scene_${DATE}"
SCENE="scripts_mix_it_dens_cd_cf/box_scene_3phase_datasets_20260111.json"

# 不带vf残差
YAML_FILE="/workspace/xyh_synology/graduate/weights/mix_it_dens_cd_cf_separate_pos_phase_v7/20260425165521/training_config.yaml"
WEIGHTS="/workspace/xyh_synology/graduate/weights/mix_it_dens_cd_cf_separate_pos_phase_v7/20260425165521/checkpoints/ckpt-46000.index"

# 检查输入参数
if [ "$#" -ge 1 ]; then
  GPU_ID="$1"  # 如果提供了参数，将其作为GPU ID
fi

# 检查脚本是否存在
if [ ! -f "$TRAIN_SCRIPT" ]; then
  echo "错误: 脚本 '$TRAIN_SCRIPT' 未找到。"
  exit 1
fi

if [ ! -f "$RUN_SCRIPT" ]; then
  echo "错误: 脚本 '$RUN_SCRIPT' 未找到。"
  exit 1
fi

# 运行 Python 脚本，重定向输出和错误
nohup python "$RUN_SCRIPT" \
  --scene "$SCENE" \
  --output "$OUTPUT" \
  --num_steps "$NUMSTEPS" \
  --cfg "$YAML_FILE" \
  --gpu "$GPU_ID" \
  --weights "$WEIGHTS" \
  --write-ply \
  "$TRAIN_SCRIPT" > "$LOG_FILE" 2>&1 &

PID=$!
echo "脚本已在后台运行，进程 ID: $PID"
echo "脚本已在后台运行，日志输出到 $LOG_FILE"
echo "可以使用以下命令查看训练进度:"
echo "  tail -f $LOG_FILE"
exit 0