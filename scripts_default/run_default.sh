#!/bin/bash

# 定义变量, 批量处理文件夹
RUN_SCRIPT="scripts/run_network.py"
TRAIN_SCRIPT="scripts/train_network_tf.py"
# YAML_FILE="scripts_default/default_snas4.yaml"
DATE=$(date +"%Y%m%d_%H%M%S")  # 获取当前日期和时间，格式为 YYYYMMDD_HHMMSS
LOG_FILE="scripts_default/run_default_${DATE}.log"  # 定义日志文件名，包含日期和时间
SCENE="scripts_default/example_scene.json"
OUTPUT="/workspace/xyh_synology/graduate/run_network/cconv_default/ply_${DATE}"
NUMSTEPS=400
GPU_ID="0"  # 默认使用GPU 0
# WEIGHTS="/workspace/xyh_synology/graduate/weights/cconv_deafault/train_network_tf_default_snas4_2025_04_23/checkpoints/ckpt-5000.index"
WEIGHTS="/workspace/xyh_synology/graduate/weights/cconv_deafault/train_network_tf_default_snas4_2025_04_23/checkpoints/ckpt-1000.index"
# WEIGHTS="/workspace/DeepLagrangianFluids/scripts/pretrained_model_weights.h5"

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
  --weights "$WEIGHTS" \
  --write-ply \
  "$TRAIN_SCRIPT" > "$LOG_FILE" 2>&1 &

PID=$!
echo "脚本已在后台运行，进程 ID: $PID"
echo "脚本已在后台运行，日志输出到 $LOG_FILE"
echo "可以使用以下命令查看模拟进度:"
echo "  tail -f $LOG_FILE"
exit 0