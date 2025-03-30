#!/bin/bash

# 定义变量, 批量处理文件夹
TRAIN_SCRIPT="train_network_mixfluid_tf.py"
YAML_FILE="mix-fluid.yaml"
DATE=$(date +"%Y%m%d_%H%M%S")  # 获取当前日期和时间，格式为 YYYYMMDD_HHMMSS
LOG_FILE="train_mix_fluid_${DATE}.log"  # 定义日志文件名，包含日期和时间

# 检查脚本是否存在
if [ ! -f "$TRAIN_SCRIPT" ]; then
  echo "错误: 脚本 '$TRAIN_SCRIPT' 未找到。"
  exit 1
fi

# 运行 Python 脚本，重定向输出和错误
nohup python "$TRAIN_SCRIPT" "$YAML_FILE" > "$LOG_FILE" 2>&1 &

echo "脚本已在后台运行，日志输出到 $LOG_FILE"
exit 0