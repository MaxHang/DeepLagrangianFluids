#!/bin/bash

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONIOENCODING=utf-8

# ====== 配置区 ======
SCENE_ID="cd_0.35_cf_0.7"

LABELS=(
"CConv基线"
"Datasets变体"
"本文模型"
)

FILES=(
"/workspace/xyh_synology/graduate/weights/mix_it_dens_cd_cf_separate_pos_phase_v5/20251031150706/checkpoints/ckpt-100000.index_eval_metrics.json"
"/workspace/xyh_synology/graduate/weights/mix_separate_pos_phase/train_network_mix_tf_spearate_pos_phase_mix_separate_pos_phase_20250616/model_weights_2025_06_17.h5_eval_metrics.json"
"/workspace/xyh_synology/graduate/weights/mix-fluid/train_network_mix_tf_mix-fluid_20250507/model_weights_2025_05_08.h5_eval_metrics.json"
)

# ====== 四个指标 ======
METRICS=(
"position_mse"
"vf_mse"
"mass_drift_per_phase"
"kinetic_energy"
)

# ====== 循环执行 ======
for METRIC in "${METRICS[@]}"
do
    echo "📊 正在绘制: $METRIC"

    python compare_models_20260302_zh.py \
    --scene_id "$SCENE_ID" \
    --metric "$METRIC" \
    --labels "${LABELS[@]}" \
    -- \
    "${FILES[@]}"

    echo "✅ 完成: $METRIC"
    echo "-----------------------------------"
done