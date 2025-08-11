# flexible_multi_phase_model_v1.py

"""
灵活的多相流体粒子网络 - V1基础版

文件目的：
本文件定义了一个支持可变相数和动态物理属性的多相流预测器模型。
它是对原始代码的第一次迭代改进，专注于两个核心增强：
1. 引入了密度作为物理特征。
2. 使模型能够处理不同数量的相。

版本特点 (V1-Flexible):
1.  **引入密度 (Density)**: 这是最关键的物理改进。模型在每次调用时接受一个'phase_densities'列表，
    并根据粒子的体积分数计算'混合密度'，将其作为输入特征。这使得网络有能力学习
    与密度相关的基本物理现象（如重力分层）。

2.  **支持可变相数 (Variable Number of Phases)**: 模型不再局限于一个固定的相数。
    - **最大相数 (`max_num_phases`)**: 模型在初始化时定义一个能处理的最大相数，所有与相数相关的
      网络层都据此构建。
    - **动态调用**: 在`call`方法中，模型接收当前的实际相数`current_num_phases`。
    - **填充与掩码 (Padding & Masking)**: 模型内部自动处理输入（体积分数）的填充和
      输出（体积分数 logits）的掩码，以适应不同的相数。

3.  **动态物理属性**: 流体的物理属性（密度）不再是模型初始化时的固定参数，而是作为`call`
    方法的输入。这意味着每个训练样本都可以有自己独特的流体密度组合，极大地增强了
    模型的泛化能力。

与更高级版本的区别：
- 本版本未包含粘度特征。
- 本版本未包含显式的浮力积分器。
- 本版本未包含用于体积守恒的密度损失函数（该损失需在训练循环中另外实现）。
它是一个专注于核心功能、简洁且灵活的起点。
"""

import tensorflow as tf
import open3d.ml.tf as ml3d
import numpy as np
from debug_utils import debug_print


class MultiPhaseParticleNetwork(tf.keras.Model):

    def __init__(self,
                 # --- 核心改动：使用 max_num_phases ---
                 num_phases=2, # 模型能处理的最大相数
                 # ------------------------------------
                 kernel_size=[4, 4, 4],
                 radius_scale=1.5,
                 coordinate_mapping='ball_to_cube_volume_preserving',
                 interpolation='linear',
                 use_window=True,
                 particle_radius=0.05,
                 timestep=1 / 50,
                 gravity=(0, -9.81, 0),
                 cd_cf_as_input=True,
                 cd_cf_embedding_dim=16):
        super().__init__(name=type(self).__name__)

        self.num_phases = max_num_phases

        self.shared_feature_channels = [32, 64, 64]
        self.pos_output_channels = 3
        # 体积分数(VF)的输出通道数固定为最大值
        self.vf_output_channels = self.num_phases

        self.kernel_size = kernel_size
        self.radius_scale = radius_scale
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.use_window = use_window
        self.particle_radius = particle_radius
        self.filter_extent = np.float32(self.radius_scale * 6 * self.particle_radius)
        self.timestep = timestep
        self.gravity = tf.constant(gravity, dtype=tf.float32)

        self.cd_cf_as_input = cd_cf_as_input
        self.cd_cf_embedding_dim = cd_cf_embedding_dim

        debug_print(f"Model initialized to handle up to {self.num_phases} phases.")

        if self.cd_cf_as_input and self.cd_cf_embedding_dim > 0:
            self.cd_embedding_layer = tf.keras.layers.Dense(self.cd_cf_embedding_dim, activation='tanh', name='cd_embedding')
            self.cf_embedding_layer = tf.keras.layers.Dense(self.cd_cf_embedding_dim, activation='tanh', name='cf_embedding')

        self._all_convs = []

        def window_poly6(r_sqr):
            r_sqr_clipped = tf.maximum(r_sqr, 0.0)
            return tf.clip_by_value((1 - r_sqr_clipped)**3, 0, 1)

        def Conv(name, activation=None, **kwargs):
            conv_fn = ml3d.layers.ContinuousConv
            window_fn = window_poly6 if self.use_window else None
            conv = conv_fn(name=name, 
                           kernel_size=self.kernel_size, 
                           activation=activation, 
                           align_corners=True,
                           interpolation=self.interpolation, 
                           coordinate_mapping=self.coordinate_mapping,
                           normalize=False, 
                           window_function=window_fn, 
                           radius_search_ignore_query_points=True, 
                           **kwargs)
            self._all_convs.append((name, conv))
            return conv

        # 定义网络层
        self.shared_conv0_fluid = Conv(name="shared_conv0_fluid", filters=self.shared_feature_channels[0], activation=None)
        self.shared_conv0_obstacle = Conv(name="shared_conv0_obstacle", filters=self.shared_feature_channels[0], activation=None)
        self.shared_dense0_fluid = tf.keras.layers.Dense(name="shared_dense0_fluid", units=self.shared_feature_channels[0], activation=None)

        self.shared_convs = []
        self.shared_denses = []
        for i in range(1, len(self.shared_feature_channels)):
            ch = self.shared_feature_channels[i]
            self.shared_denses.append(tf.keras.layers.Dense(units=ch, name=f"shared_dense{i}", activation=None))
            self.shared_convs.append(Conv(name=f"shared_conv{i}", filters=ch, activation=None))

        self.pos_final_conv = Conv(name="pos_final_conv", filters=self.pos_output_channels, activation=None)
        self.pos_final_dense = tf.keras.layers.Dense(units=self.pos_output_channels, name="pos_final_dense", activation=None)

        # 只有在需要处理多相流时才创建VF预测层
        if self.num_phases > 1:
            self.vf_final_conv = Conv(name="vf_final_conv", filters=self.vf_output_channels, activation=None)
            self.vf_final_dense = tf.keras.layers.Dense(units=self.vf_output_channels, name="vf_final_dense", activation=None)

    def call(self, inputs, current_num_phases, phase_densities, training=False, fixed_radius_search_hash_table=None, cd=0.5, cf=0.5):
        """
        模型的前向传播。
        Args:
            inputs (tuple): (pos1, vel1, current_phase_fractions, box_pos, box_feats)
            current_num_phases (tf.Tensor): 标量整数, 当前批次的实际相数。
            phase_densities (tf.Tensor or list): 当前激活相的密度列表/张量。
        """
        pos1, vel1, current_phase_fractions, box_pos, box_feats = inputs
        
        phase_densities = tf.convert_to_tensor(phase_densities, dtype=tf.float32)

        # debug_print(f"pos 100: {pos1[::100].numpy()}")
        # 1. 积分位置和速度（基础版本）
        pos2_integrated, vel2_integrated = self.integrate_pos_vel(pos1, vel1)
        # debug_print(f"pos2_integrated 100: {pos2_integrated[::100].numpy()}")

        # 2. 计算共享特征
        shared_features = self.compute_shared_features(
            pos2_integrated, 
            vel2_integrated, 
            current_phase_fractions, 
            current_num_phases,
            phase_densities, # 传入动态密度
            box_pos, 
            box_feats, 
            fixed_radius_search_hash_table, 
            cd_scalar=cd, 
            cf_scalar=cf)

        # 3. 基于共享特征预测位置修正
        pos_correction = self.compute_position_correction_from_shared(shared_features, pos2_integrated)

        # 4. 应用位置修正并计算最终速度
        pos_final, vel_final = self.compute_new_pos_vel(pos1, vel1, pos2_integrated, vel2_integrated, pos_correction)

        # 5. 基于共享特征预测相分数logits
        next_phase_fractions_final = current_phase_fractions
        if self.num_phases > 1 and current_num_phases > 1:
            # a. 预测所有 max_num_phases 的 logits
            next_vf_logits_padded = self.compute_vf_logits_from_shared(shared_features, pos_final)
            
            # b. 使用掩码计算下一时刻的体积分数
            next_phase_fractions_padded = self.compute_next_phase_fractions(
                next_vf_logits_padded, current_num_phases)
            
            # c. 切片，只保留有效的相，得到最终输出
            next_phase_fractions_final = next_phase_fractions_padded[:, :current_num_phases]
        
        return pos_final, vel_final, next_phase_fractions_final

    def compute_shared_features(self, 
                                pos, 
                                vel, 
                                phase_fractions, 
                                current_num_phases,
                                phase_densities, 
                                box_pos, 
                                box_feats,
                                fixed_radius_search_hash_table=None, 
                                cd_scalar=0.5, 
                                cf_scalar=0.5):
        filter_extent_tensor = tf.constant(self.filter_extent, dtype=tf.float32)

        # --- 输入填充 (Input Padding) ---
        # `phase_fractions` 的形状为 [N, current_num_phases]，需要填充到 [N, max_num_phases]
        padding_size = self.num_phases - current_num_phases
        # `paddings` 的格式是 [[dim1_before, dim1_after], [dim2_before, dim2_after]]
        paddings = [[0, 0], [0, padding_size]]
        padded_phase_fractions = tf.pad(phase_fractions, paddings, "CONSTANT", constant_values=0)

        # 计算混合密度
        mixture_density = tf.reduce_sum(phase_fractions * phase_densities, axis=-1, keepdims=True)

        # debug_print(f"mixture_density shape: {mixture_density.shape}")
        # debug_print(f"Current densities 100: {mixture_density[::100].numpy()}")

        fluid_feats_list = [
            tf.ones_like(pos[:, 0:1]),   # 1. 存 njm mj征
            vel,                         # 2. 当前速度
            mixture_density,             # 3. (新增) 混合密度
            padded_phase_fractions       # 4. (填充后) 各相体积分数
        ]

        if self.cd_cf_as_input:
            cd_tensor_val = tf.cast(cd_scalar, dtype=tf.float32)
            cf_tensor_val = tf.cast(cf_scalar, dtype=tf.float32)
            batch_size = tf.shape(pos)[0]
            if self.cd_cf_embedding_dim > 0:
                cd_embed = self.cd_embedding_layer(tf.ones((batch_size, 1)) * cd_tensor_val)
                cf_embed = self.cf_embedding_layer(tf.ones((batch_size, 1)) * cf_tensor_val)
                fluid_feats_list.extend([cd_embed, cf_embed])
            else:
                fluid_feats_list.extend([tf.ones_like(pos[:, 0:1]) * cd_tensor_val,
                                         tf.ones_like(pos[:, 0:1]) * cf_tensor_val])

        fluid_feats = tf.concat(fluid_feats_list, axis=-1)
        debug_print("Shape of fluid_feats for shared features: ", tf.shape(fluid_feats))
        
        # --- 网络前向传播 ---
        shared_conv0_fluid = self.shared_conv0_fluid(fluid_feats, pos, pos, filter_extent_tensor)
        shared_dense0_fluid = self.shared_dense0_fluid(fluid_feats)
        shared_conv0_obstacle = self.shared_conv0_obstacle(box_feats, box_pos, pos, filter_extent_tensor)
        
        processed_feats = tf.concat([shared_conv0_obstacle, shared_conv0_fluid, shared_dense0_fluid], axis=-1)

        shared_ans_convs = [processed_feats]
        for conv_layer, dense_layer in zip(self.shared_convs, self.shared_denses):
            current_features = tf.keras.activations.relu(shared_ans_convs[-1])
            ans_conv = conv_layer(current_features, pos, pos, filter_extent_tensor)
            ans_dense = dense_layer(current_features)
            
            if ans_dense.shape[-1] == shared_ans_convs[-1].shape[-1]:
                ans = ans_conv + ans_dense + shared_ans_convs[-1] # 残差连接
            else:
                ans = ans_conv + ans_dense
            shared_ans_convs.append(ans)

        # 存储邻居数量信息（用于损失计算）
        self.num_fluid_neighbors = ml3d.ops.reduce_subarrays_sum(
            tf.ones_like(self.shared_conv0_fluid.nns.neighbors_index, dtype=tf.float32),
            self.shared_conv0_fluid.nns.neighbors_row_splits)

        return shared_ans_convs[-1]

    def compute_next_phase_fractions(self, network_vf_logits_padded, current_num_phases):
        """
        使用掩码来计算下一时刻的体积分数，确保概率只在有效相之间分配。
        Args:
            network_vf_logits_padded (tf.Tensor): [N, max_num_phases] 来自网络的原始输出
            current_num_phases (tf.Tensor): 当前有效的相数
        """
        # 创建一个掩码，有效部分为0，无效部分为一个很大的负数
        # 形状为 [max_num_phases]
        mask_range = tf.range(self.num_phases, dtype=tf.int32)
        # tf.where(condition, value_if_true, value_if_false)
        mask = tf.where(mask_range < current_num_phases, 0.0, -1e9)
        
        # 将掩码应用到 logits 上。广播机制会自动将 [max_num_phases] 的掩码加到 [N, max_num_phases] 的logits上
        masked_logits = network_vf_logits_padded + mask
        
        # Softmax 会自动处理 -1e9，使其对应的概率接近于0
        next_fractions_padded = tf.nn.softmax(masked_logits, axis=-1)
        
        return next_fractions_padded
    
    # --- 辅助方法（与原始代码或V1-V3版本一致） ---

    def integrate_pos_vel(self, pos1, vel1):
        dt = self.timestep
        vel2 = vel1 + dt * self.gravity
        pos2 = pos1 + dt * (vel1 + vel2) / 2.0
        return pos2, vel2

    def compute_new_pos_vel(self, pos1, vel1, pos2_integrated, vel2_integrated, pos_correction):
        dt = self.timestep
        pos_final = pos2_integrated + pos_correction
        vel_final = (pos_final - pos1) / dt
        return pos_final, vel_final

    def compute_position_correction_from_shared(self, shared_features, pos):
        filter_extent_tensor = tf.constant(self.filter_extent, dtype=tf.float32)
        current_features = tf.keras.activations.relu(shared_features)
        pos_conv = self.pos_final_conv(current_features, pos, pos, filter_extent_tensor)
        pos_dense = self.pos_final_dense(current_features)
        pos_output = pos_conv + pos_dense
        pos_correction = (1.0 / 128.0) * pos_output
        debug_print("Shape of position correction: ", tf.shape(pos_correction))
        return pos_correction

    def compute_vf_logits_from_shared(self, shared_features, pos):
        if self.num_phases <= 1:
            return None
        filter_extent_tensor = tf.constant(self.filter_extent, dtype=tf.float32)
        current_features = tf.keras.activations.relu(shared_features)
        vf_conv = self.vf_final_conv(current_features, pos, pos, filter_extent_tensor)
        vf_dense = self.vf_final_dense(current_features)
        vf_logits = vf_conv + vf_dense
        debug_print("Shape of vf_logits from shared network: ", tf.shape(vf_logits))
        return vf_logits
    
    def init(self, feats_shape=None):
        """
        使用虚拟数据初始化模型，以构建网络权重并打印摘要。
        此方法适配了模型的灵活性，使用一个典型的场景（2相流）来完成初始化。
        """
        # 定义一个典型的初始化场景
        init_num_phases = tf.constant(2)
        
        # 确保虚拟相数不超过模型支持的最大相数
        if init_num_phases > self.num_phases:
            raise ValueError(f"Initialization phase count ({init_num_phases}) cannot exceed max_num_phases ({self.num_phases}).")

        # 创建符合该场景的虚拟输入数据
        pos = np.zeros(shape=(1, 3), dtype=np.float32)
        vel = np.zeros(shape=(1, 3), dtype=np.float32)
        
        # 体积分数张量的维度由 init_num_phases 决定
        phase_fractions = np.zeros(shape=(1, init_num_phases.numpy()), dtype=np.float32)
        phase_fractions[:, 0] = 1.0

        box = np.zeros(shape=(1, 3), dtype=np.float32)
        box_feats = np.zeros(shape=(1, 3), dtype=np.float32)
        
        # 创建匹配的虚拟物理属性
        init_densities = np.ones(shape=(init_num_phases.numpy(),), dtype=np.float32) * 1000.0

        cd = np.float32(0.5)
        cf = np.float32(0.5)
        
        # 使用适配后的 call 方法签名进行调用
        _ = self.__call__((pos, vel, phase_fractions, box, box_feats),
                          current_num_phases=init_num_phases,
                          phase_densities=init_densities,
                          cd=cd, cf=cf)
        
        print(f"{self.name} initialized to handle up to {self.num_phases} phases (tested with {init_num_phases.numpy()} phases).")
        print(f"Shared feature channels: {self.shared_feature_channels}")
        print(f"Position output channels: {self.pos_output_channels}")
        if self.num_phases > 1:
            print(f"VF output channels (max): {self.vf_output_channels}")
        
        try:
            # 打印模型摘要
            self.summary()
        except Exception as e:
            print(f"Could not print model summary: {e}")


# --- 如何使用这个灵活的模型 ---
#
# # 1. 初始化模型时，只需指定最大相数
# model = FlexibleMultiPhaseModel_V1(num_phases=2)
#
# # 2. 准备一个2相流的训练样本
# num_phases_2 = tf.constant(2)
# densities_2 = [1000.0, 500.0] # 水和油
# # 假设 inputs_2_phase 是一个包含 (pos, vel, vf, box_pos, box_feats) 的元组
# # 其中 vf 的形状是 [num_particles, 2]
# # pred_pos, pred_vel, pred_vf_2 = model(inputs_2_phase,
# #                                      current_num_phases=num_phases_2,
# #                                      phase_densities=densities_2)
# # pred_vf_2 的形状将是 [num_particles, 2]
#
# # 3. 准备一个3相流的训练样本
# num_phases_3 = tf.constant(3)
# densities_3 = [1000.0, 13600.0, 800.0] # 水、水银、油
# # 假设 inputs_3_phase 的体积分数 vf 的形状是 [num_particles, 3]
# # pred_pos, pred_vel, pred_vf_3 = model(inputs_3_phase,
# #                                      current_num_phases=num_phases_3,
# #                                      phase_densities=densities_3)
# # pred_vf_3 的形状将是 [num_particles, 3]