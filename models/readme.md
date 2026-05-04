## mix模型演变记录
- v1: 采用连续卷积主干，但使用了一个简化的多相信息处理方式：直接将前 max_num_phases 个相的体积分数和原始密度值拼接作为输入（不足则补零），且VF预测头直接预测下一时刻的绝对值。
- v2: 在v1基础上
    - 引入多相特征编码器来处理可变相数(可以选择聚合方式mean or sum)
    - 加入我们提出的对数中心化与缩放特征归一化方法
    - 将VF预测头替换为我们在概率空间进行残差更新与显式归一化的机制。
- v3: 在v2基础上
    - 修改compute_delta_vf_from_shared，让它只预测 N-1 个相的变化量，然后通过数学计算得出最后一个，保证 Σ(ΔVF) = 0。这将从架构层面**强制守恒，是解决这个问题的最根本方法。
- v5
    - 真正的多相流模型应该对不同相数使用相应维度
    - 共享主干（80-90%参数）可以学到通用的流体物理规律
    - VF预测头（10-20%参数）是相数特定的，vf_heads[5]
    - 推荐方案：多阶段迁移学习
        - 阶段1：在2相数据上预训练 # 训练充分，让共享主干学好流体物理
        - 使用少量5相标注数据 只需要少量5相数据（10-20%） 即可训练 vf_heads[5]
            ```python
            for layer in model.layers:
                if 'shared' in layer.name or 'phase_encoder' in layer.name:
                    layer.trainable = False  # 冻结主干
            ```
    - 其他可选方案
        - 参数共享的VF预测头（需要修改架构）
            - 修改 V5 模型，让所有相数共享一个预测头
            - 输出固定维度 max_num_phases，运行时截取前 current_num_phases 个


- v12 相比 v11，只想训练脚本加了importance，以及loss_weights



## deepset encoder 演变记录
- 初始
    - 输入特征需要特定的最大相维度（需要padding），比如[N, max_num_phases, num_phase_features]
- v2 
    - 考虑到相数通常只有2-5个
    - 使用tf.map_fn 循环替代 TimeDistributed + Padding 的批处理
    - 这样输入的维度为 [N, num_phases, num_phase_features]


## nomix 模型
- default_tf_nomix.py
