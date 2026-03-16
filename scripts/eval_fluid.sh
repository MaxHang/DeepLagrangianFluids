#!/bin/bash

################################################

## 这个脚本用于评估 Nomix Fluid 模型的性能。它会运行指定的 Python 脚本，并将输出日志保存到一个包含当前日期和时间的文件中。用户可以通过提供 GPU ID 来指定使用哪个 GPU 进行评估。

## 这里使用单相流体的 model   weights

################################################


# 定义变量, 批量处理文件夹
EVAL_SCRIPT="scripts/evaluate_nomix_network_by_single_model.py"
TRAIN_SCRIPT="scripts/train_network_tf.py"
YAML_FILE="scripts/single_eval_nomix-fluid.yaml"
DATE=$(date +"%Y%m%d_%H%M%S")  # 获取当前日期和时间，格式为 YYYYMMDD_HHMMSS
LOG_FILE="scripts/log_eval/single_eval_nomix_fluid_${DATE}.log"  # 定义日志文件名，包含日期和时间
GPU_ID="0"  # 默认使用GPU 0
WEIGHTS="scripts/model_weights.h5"

# 检查输入参数
if [ "$#" -ge 1 ]; then
  GPU_ID="$1"  # 如果提供了参数，将其作为GPU ID
fi

# 检查脚本是否存在
if [ ! -f "$TRAIN_SCRIPT" ]; then
  echo "错误: 脚本 '$TRAIN_SCRIPT' 未找到。"
  exit 1
fi

if [ ! -f "$EVAL_SCRIPT" ]; then
  echo "错误: 脚本 '$EVAL_SCRIPT' 未找到。"
  exit 1
fi

# 运行 Python 脚本，重定向输出和错误
nohup python "$EVAL_SCRIPT" \
  --trainscript "$TRAIN_SCRIPT" \
  --cfg "$YAML_FILE" \
  --weights "$WEIGHTS"  > "$LOG_FILE" 2>&1 &

PID=$!
echo "脚本已在后台运行，进程 ID: $PID"
echo "脚本已在后台运行，日志输出到 $LOG_FILE"
echo "可以使用以下命令查看训练进度:"
echo "  tail -f $LOG_FILE"
exit 0