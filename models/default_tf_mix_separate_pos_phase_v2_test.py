# flexible_multi_phase_model_v1.py

"""
灵活的多相流体粒子网络 - V2基础版

版本特点 (V2 compared to V1):
1.  **支持可变相数 (Variable Number of Phases)**: 模型不再局限于一个固定的相数。
    将 DeepSetPhaseEncoder 集成到主网络中

3.  **残差预测 (Residual Prediction) **: 流体的物理属性（密度）不再是模型初始化时的固定参数，而是作为`call`
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
import time

class DeepSetPhaseEncoder(tf.keras.Model):
    """
    使用 Deep Sets 架构将一个粒子上可变数量的相信息编码为固定大小的特征向量。
    """
    def __init__(self, phi_dims, rho_dims, name="DeepSetPhaseEncoder"):
        """
        Args:
            phi_dims (list of int): φ 网络的每个隐藏层和输出层的维度。
            rho_dims (list of int): ρ 网络的每个隐藏层和输出层的维度。
        """
        super().__init__(name=name)
        
        # φ 网络：处理单个相的特征
        self.phi_net = tf.keras.Sequential(name="phi_network")
        for dim in phi_dims:
            self.phi_net.add(tf.keras.layers.Dense(dim, activation='relu'))
        
        # ρ 网络：处理聚合后的特征
        self.rho_net = tf.keras.Sequential(name="rho_network")
        for dim in rho_dims:
            self.rho_net.add(tf.keras.layers.Dense(dim, activation='relu'))
            
    def call(self, inputs):
        """
        Args:
            inputs (tuple): 包含两个张量
                - phase_features (tf.Tensor): 形状为 [N, max_num_phases, num_phase_features]
                  N 通常是 batch_size * num_particles
                - mask (tf.Tensor): 形状为 [N, max_num_phases]，布尔或浮点型掩码，
                  标记哪些是真实的相（1.0）哪些是填充（0.0）。
        
        Returns:
            tf.Tensor: 形状为 [N, rho_output_dim]，代表每个粒子的固定维度嵌入。
        """
        phase_features, mask = inputs
        
        # 1. 应用 φ 网络到每个相
        #    输入: [N, max_num_phases, num_phase_features]
        #    输出: [N, max_num_phases, phi_output_dim]
        # TimeDistributed 会将 phi_net 应用到时间步（这里是 max_num_phases 维度）上
        phi_output = tf.keras.layers.TimeDistributed(self.phi_net)(phase_features)
        
        # 2. 应用掩码，将填充部分的输出置零，以免影响求和
        #    我们需要将掩码扩展一个维度以进行广播
        mask_expanded = mask[..., tf.newaxis] # 形状变为 [N, max_num_phases, 1]
        masked_phi_output = phi_output * tf.cast(mask_expanded, dtype=phi_output.dtype)
        
        # 3. 聚合（置换不变操作）
        #    沿 'max_num_phases' 维度求和
        #    输出: [N, phi_output_dim]
        aggregated_features = tf.reduce_sum(masked_phi_output, axis=1)
        
        # 4. 应用 ρ 网络
        #    输入: [N, phi_output_dim]
        #    输出: [N, rho_output_dim]
        final_embedding = self.rho_net(aggregated_features)
        
        return final_embedding

# 为了独立运行，我们先定义一个简单的 debug_print
def debug_print(*args, **kwargs):
    print(*args, **kwargs) # 在需要时取消注释
    pass


class MultiPhaseParticleNetwork(tf.keras.Model):

    def __init__(self,
                 # --- 核心改动：使用 max_num_phases ---
                 max_num_phases=5, # 模型能处理的最大相数
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

        self.max_num_phases = max_num_phases

        self.shared_feature_channels = [64, 64, 64]
        self.pos_output_channels = 3
        # 体积分数(VF)的输出通道数固定为最大值
        self.vf_output_channels = self.max_num_phases

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

        debug_print(f"Model initialized to handle up to {self.max_num_phases} phases.")

        if self.cd_cf_as_input and self.cd_cf_embedding_dim > 0:
            self.cd_embedding_layer = tf.keras.layers.Dense(self.cd_cf_embedding_dim, activation='tanh', name='cd_embedding')
            self.cf_embedding_layer = tf.keras.layers.Dense(self.cd_cf_embedding_dim, activation='tanh', name='cf_embedding')

        # --- 在主网络中定义批归一化层 ---
        self.phase_feature_layer_norm = tf.keras.layers.LayerNormalization(name="phase_feature_ln")

        # 创建 Deep Sets 编码器实例
        self.phase_encoder = DeepSetPhaseEncoder(
            phi_dims=[64, 128],  # 示例维度
            rho_dims=[128, 64]   # 示例维度
        )

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
        if self.max_num_phases > 1:
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
            cf_scalar=cf,
            training=training)

        # 3. 基于共享特征预测位置修正
        pos_correction = self.compute_position_correction_from_shared(shared_features, pos2_integrated)

        # 4. 应用位置修正并计算最终速度
        pos_final, vel_final = self.compute_new_pos_vel(pos1, vel1, pos2_integrated, vel2_integrated, pos_correction)

        # 5. 基于共享特征预测相分数logits
        next_phase_fractions_final = current_phase_fractions
        if self.max_num_phases > 1 and current_num_phases > 1:
            # a. 预测体积分数的变化量 (delta_logits)
            #    网络输出的含义已变为“修正量”，但函数名可以保持不变。
            predicted_delta_logits = self.compute_vf_logits_from_shared(shared_features, pos_final)
            
            # b. 准备残差连接所需的、经过填充的当前体积分数
            padding_size = self.max_num_phases - current_num_phases
            paddings = [[0, 0], [0, padding_size]] # 假设VF形状是 [N, current_num_phases]
            vf_current_padded = tf.pad(current_phase_fractions, paddings, "CONSTANT", constant_values=0)
            
            # c. 调用新的残差计算函数
            vf_next_padded = self.compute_next_phase_fractions(
                vf_current_padded, 
                predicted_delta_logits,
                current_num_phases
            )
            
            # d. 切片，得到最终的、符合当前相数的输出 (这一步仍然是必要的)
            next_phase_fractions_final = vf_next_padded[:, :current_num_phases]
        
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
                                cf_scalar=0.5,
                                training=False):
        start_time = time.time()
        # phase_fractions 形状 [N, num_particles, current_num_phases]
        # phase_densities 形状 [current_num_phases]

        # --- 1. 准备 Deep Sets 的输入 ---
        # a. 创建每个相的特征向量 [vf, density]
        #    我们需要为每个粒子构建这个集合
        
        # 将 phase_densities 广播到每个粒子上
        # 形状: [N, num_particles, current_num_phases]
        densities_per_particle = tf.broadcast_to(phase_densities, tf.shape(phase_fractions))
        debug_print("Shape of densities_per_particle: ", tf.shape(densities_per_particle))

        # b. 进行特定于数据的预处理
        # b.1. 对数变换密度
        log_densities_per_particle = tf.math.log(densities_per_particle + 1e-8)
        
        # b.2. 拼接成完整的相特征向量 [vf, log_density]
        per_phase_features = tf.stack([phase_fractions, log_densities_per_particle], axis=-1)

        # b,3. 应用归一化
        normalized_features = self.phase_feature_layer_norm(per_phase_features, training=training)
        debug_print("Shape of normalized_features: ", tf.shape(normalized_features))
        
        # c. 进行填充以匹配 max_num_phases
        padding_size = self.max_num_phases - current_num_phases
        # paddings 格式: [[axis0_pad], [axis1_pad], [axis2_pad], [axis3_pad]]
        paddings = [[0, 0], [0, padding_size], [0, 0]]
        padded_features = tf.pad(normalized_features, paddings, "CONSTANT")
        
        # d. 创建掩码
        mask_range = tf.range(self.max_num_phases)
        mask = mask_range < current_num_phases # 形状: [max_num_phases]
        # 广播到每个粒子上
        mask_per_particle = tf.broadcast_to(mask, [tf.shape(pos)[0], self.max_num_phases])

        debug_print("Shape of padded_features: ", tf.shape(padded_features))
        debug_print("Shape of mask_per_particle: ", tf.shape(mask_per_particle))

        # --- 2. 使用 Deep Sets 编码器 ---
        deep_sets_start = time.time()
        particle_phase_embedding = self.phase_encoder([padded_features, mask_per_particle])
        deep_sets_time = time.time() - deep_sets_start

        filter_extent_tensor = tf.constant(self.filter_extent, dtype=tf.float32)

        fluid_feats_list = [
            tf.ones_like(pos[:, 0:1]),   # 1. 存在特征
            vel,                         # 2. 当前速度
            particle_phase_embedding,    # 3. 来自 Deep Sets 的多相嵌入
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
        
        total_time = time.time() - start_time

        # if training and tf.random.uniform([]) < 0.01:  # 1%的概率打印时间
        #     tf.print(f"Deep Sets time: {deep_sets_time:.4f}s, Total: {total_time:.4f}s")
        tf.print(f"Deep Sets time: {deep_sets_time:.4f}s, Total: {total_time:.4f}s")

        return shared_ans_convs[-1]
    
    # 再次提供最关键的修复代码，请务必替换
    def compute_next_phase_fractions(self, current_phase_fractions_padded, delta_logits_padded, current_num_phases):
        # 1. 创建只在有效区域为 1.0 的掩码
        mask_range = tf.range(self.max_num_phases, dtype=tf.int32)
        phase_mask = mask_range < current_num_phases
        phase_mask_float = tf.cast(tf.broadcast_to(phase_mask, tf.shape(current_phase_fractions_padded)), dtype=tf.float32)

        # 2. 安全地计算 current_logits
        vf_safe = current_phase_fractions_padded + 1e-8
        log_vf = tf.math.log(vf_safe)
        current_logits = log_vf * phase_mask_float # 只保留有效区域的log值

        # 3. 安全地计算 next_logits
        next_logits_padded = current_logits + (delta_logits_padded * phase_mask_float) # 只在有效区域加delta

        # 4. 应用 Softmax 掩码
        softmax_mask = tf.where(phase_mask, 0.0, -1e9)
        masked_logits = next_logits_padded + softmax_mask
        
        # 5. 应用 Softmax
        next_fractions_padded = tf.nn.softmax(masked_logits, axis=-1)
        
        # 6. 确保填充区严格为零
        final_fractions_padded = next_fractions_padded * phase_mask_float
        
        return final_fractions_padded

    # def compute_next_phase_fractions(self, current_phase_fractions_padded, delta_logits_padded, current_num_phases):
    #     """
    #     使用残差连接和Softmax来计算下一时刻的体积分数。
    #     这个版本比直接预测logits更稳定、更容易学习。

    #     Args:
    #         current_phase_fractions_padded (tf.Tensor): [N, max_num_phases], 经过填充的当前时刻体积分数。
    #         delta_logits_padded (tf.Tensor): [N, max_num_phases], 网络预测出的 "变化量" 的logits。
    #         current_num_phases (tf.Tensor): 标量整数, 当前有效的相数。

    #     Returns:
    #         tf.Tensor: [N, max_num_phases], 经过归一化的、下一时刻的、填充后的体积分数。
    #     """
    #     # --- 残差连接的核心逻辑 ---
    #     # 1. 将当前的体积分数（概率空间）转换到一个类似logit的空间。
    #     #    log(vf) 是 softmax 的逆操作的近似。加上一个小的epsilon防止log(0)。
    #     current_logits = tf.math.log(current_phase_fractions_padded + 1e-8)
        
    #     # 2. 将网络预测的变化量logits加到当前logits上（残差连接）。
    #     #    网络现在学习的是logits空间的变化量 Δ_logits。
    #     next_logits_padded = current_logits + delta_logits_padded
    #     # ---------------------------

    #     # --- 掩码与归一化 (这部分与之前版本相同) ---
    #     # 3. 创建掩码，确保概率只在有效的相之间进行归一化。
    #     mask_range = tf.range(self.max_num_phases, dtype=tf.int32)
    #     mask = tf.where(mask_range < current_num_phases, 0.0, -1e9)
        
    #     # 4. 应用掩码
    #     masked_logits = next_logits_padded + mask
        
    #     # 5. 应用Softmax得到最终的、归一化的下一时刻体积分数
    #     next_fractions_padded = tf.nn.softmax(masked_logits, axis=-1)
        
    #     return next_fractions_padded
    
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
        if self.max_num_phases <= 1:
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
        if init_num_phases > self.max_num_phases:
            raise ValueError(f"Initialization phase count ({init_num_phases}) cannot exceed max_num_phases ({self.max_num_phases}).")

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
        
        print(f"{self.name} initialized to handle up to {self.max_num_phases} phases (tested with {init_num_phases.numpy()} phases).")
        print(f"Shared feature channels: {self.shared_feature_channels}")
        print(f"Position output channels: {self.pos_output_channels}")
        if self.max_num_phases > 1:
            print(f"VF output channels (max): {self.vf_output_channels}")
        
        try:
            # 打印模型摘要
            self.summary()
        except Exception as e:
            print(f"Could not print model summary: {e}")


# --- 如何使用这个灵活的模型 ---
#
# # 1. 初始化模型时，只需指定最大相数
# model = FlexibleMultiPhaseModel_V1(max_num_phases=5)
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

# if __name__ == "__main__":
#     print("--- Starting Quick Model Verification ---")

#     # --- 1. 定义模拟参数 ---
#     max_p = 5         # 模型能处理的最大相数
#     current_p = 2     # 当前样本的实际相数
#     num_particles = 100 # 粒子数量
#     num_box_particles = 50 # 边界粒子数量

#     print(f"Model max phases: {max_p}, Current sample phases: {current_p}")

#     # --- 2. 创建模型实例 ---
#     try:
#         model = MultiPhaseParticleNetwork(max_num_phases=max_p)
#         print("Model instantiated successfully.")
#     except Exception as e:
#         print(f"Error during model instantiation: {e}")
#         exit()

#     # --- 3. 创建虚拟输入数据 ---
#     print("Generating fake input data...")
#     # a. 粒子数据
#     pos1 = tf.constant(np.random.rand(num_particles, 3), dtype=tf.float32)
#     vel1 = tf.constant(np.random.rand(num_particles, 3), dtype=tf.float32)
#     # b. 边界数据
#     box_pos = tf.constant(np.random.rand(num_box_particles, 3), dtype=tf.float32)
#     box_feats = tf.constant(np.random.rand(num_box_particles, 3), dtype=tf.float32)
#     # c. 体积分数数据 (确保每行和为1)
#     vf_raw = np.random.rand(num_particles, current_p)
#     vf_sum = np.sum(vf_raw, axis=1, keepdims=True)
#     vf1 = tf.constant(vf_raw / vf_sum, dtype=tf.float32)
#     # d. 动态物理属性
#     num_phases_tf = tf.constant(current_p, dtype=tf.int32)
#     densities_tf = tf.constant(np.random.rand(current_p) * 1000, dtype=tf.float32)

#     inputs_tuple = (pos1, vel1, vf1, box_pos, box_feats)
    
#     print(f"Input pos shape: {pos1.shape}")
#     print(f"Input vf shape: {vf1.shape}")
#     print(f"Input densities: {densities_tf.numpy()}")
    
#     # --- 4. 执行一次前向传播 ---
#     print("\nAttempting a single forward pass...")
#     try:
#         # 使用 @tf.function 包装以模拟训练环境
#         @tf.function
#         def run_forward_pass(model, inputs, num_phases, densities):
#             return model(inputs, num_phases, densities)

#         pos_final, vel_final, vf_final = run_forward_pass(model, inputs_tuple, num_phases_tf, densities_tf)
        
#         print("\nSUCCESS! Forward pass completed without errors. ---")
#         print(f"Output pos shape: {pos_final.shape}")
#         print(f"Output vel shape: {vel_final.shape}")
#         print(f"Output vf shape: {vf_final.shape}")

#         # 验证输出形状是否正确
#         assert pos_final.shape == pos1.shape
#         assert vel_final.shape == vel1.shape
#         assert vf_final.shape == vf1.shape
#         print("Output shapes are correct.")

#     except Exception as e:
#         print("\nFAILED! An error occurred during the forward pass. ---")
#         import traceback
#         traceback.print_exc()

if __name__ == "__main__":
    print("--- Starting Quick Model Verification with Training Test ---")

    # --- 1. 定义模拟参数（减小数据量） ---
    max_p = 5         # 减少最大相数
    current_p = 2     
    num_particles = 2000    # 大幅减少粒子数量
    num_box_particles = 5000  # 大幅减少边界粒子数量

    print(f"Model max phases: {max_p}, Current sample phases: {current_p}")

    # --- 2. 创建模型实例并添加错误处理 ---
    try:
        gpu_id = 1  # 使用第二个GPU（如果有的话）
        gpus = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
        print(f"Using GPU {gpu_id}")
        
        model = MultiPhaseParticleNetwork(max_num_phases=max_p, cd_cf_embedding_dim=8)  # 减小嵌入维度
        print("Model instantiated successfully.")
    except Exception as e:
        print(f"Error during model instantiation: {e}")
        exit()

    # --- 3. 创建更小的虚拟输入数据 ---
    print("Generating fake input data...")
    
    # 使用较小的随机数范围
    pos0 = tf.constant(np.random.rand(num_particles, 3) * 0.1, dtype=tf.float32)
    vel0 = tf.constant(np.random.rand(num_particles, 3) * 0.01, dtype=tf.float32)  # 更小的初始速度
    box_pos = tf.constant(np.random.rand(num_box_particles, 3) * 0.1, dtype=tf.float32)
    box_feats = tf.constant(np.random.rand(num_box_particles, 3) * 0.1, dtype=tf.float32)
    
    # 体积分数数据 - 创建两个时间步
    vf_raw0 = np.random.rand(num_particles, current_p)
    vf_sum0 = np.sum(vf_raw0, axis=1, keepdims=True)
    vf0 = tf.constant(vf_raw0 / vf_sum0, dtype=tf.float32)
    
    # 模拟真实的下一时间步数据（添加小的噪声）
    pos1_gt = pos0 + tf.random.normal(pos0.shape, mean=0.0, stddev=0.001, dtype=tf.float32)
    vel1_gt = vel0 + tf.random.normal(vel0.shape, mean=0.0, stddev=0.001, dtype=tf.float32)
    vf1_gt = vf0 + tf.random.normal(vf0.shape, mean=0.0, stddev=0.01, dtype=tf.float32)
    vf1_gt = tf.nn.softmax(vf1_gt, axis=-1)  # 确保体积分数和为1
    
    # 创建第二个时间步的真实数据
    pos2_gt = pos1_gt + tf.random.normal(pos1_gt.shape, mean=0.0, stddev=0.001, dtype=tf.float32)
    vel2_gt = vel1_gt + tf.random.normal(vel1_gt.shape, mean=0.0, stddev=0.001, dtype=tf.float32)
    vf2_gt = vf1_gt + tf.random.normal(vf1_gt.shape, mean=0.0, stddev=0.01, dtype=tf.float32)
    vf2_gt = tf.nn.softmax(vf2_gt, axis=-1)
    
    # 动态物理属性
    num_phases_tf = tf.constant(current_p, dtype=tf.int32)
    densities_tf = tf.constant([1000.0, 3000.0], dtype=tf.float32)  # 使用归一化的密度值

    print(f"Input shapes - pos: {pos0.shape}, vf: {vf0.shape}")
    
    # --- 4. 定义损失函数 ---
    def simple_loss_fn(pred_pos, gt_pos, pred_vf, gt_vf, current_num_phases, num_fluid_neighbors=None):
        """简化的损失函数"""
        # 位置损失
        pos_loss = tf.reduce_mean(tf.square(pred_pos - gt_pos))
        
        # 相分数损失（只在多相情况下计算）
        if current_num_phases > 1 and pred_vf is not None and gt_vf is not None:
            vf_loss = tf.reduce_mean(tf.square(pred_vf - gt_vf))
        else:
            vf_loss = 0.0
        
        total_loss = pos_loss + 0.1 * vf_loss  # vf损失权重较小
        return total_loss, pos_loss, vf_loss

    # --- 5. 创建优化器 ---
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # --- 6. 前向传播测试 ---
    print("\nStep 1: Testing forward pass...")
    try:
        inputs0 = (pos0, vel0, vf0, box_pos, box_feats)
        pos_pred1, vel_pred1, vf_pred1 = model(inputs0, num_phases_tf, densities_tf)
        
        print("SUCCESS! Forward pass completed.")
        print(f"Predicted shapes - pos: {pos_pred1.shape}, vel: {vel_pred1.shape}, vf: {vf_pred1.shape}")
        
    except Exception as e:
        print(f"Forward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # --- 7. 训练步骤测试 ---
    print("\nStep 2: Testing training step with GradientTape...")
    
    def train_step():
        """单个训练步骤"""
        with tf.GradientTape() as tape:
            # 第一次预测
            inputs1 = (pos0, vel0, vf0, box_pos, box_feats)
            pred_pos1, pred_vel1, pred_vf1 = model(
                inputs1, 
                current_num_phases=num_phases_tf,
                phase_densities=densities_tf,
                training=True,
                cd=0.5, cf=0.5
            )
            
            # 计算第一步损失
            loss1, pos_loss1, vf_loss1 = simple_loss_fn(
                pred_pos1, pos1_gt, pred_vf1, vf1_gt, current_p
            )
            
            # 第二次预测（使用第一次的预测作为输入）
            inputs2 = (pred_pos1, pred_vel1, pred_vf1, box_pos, box_feats)
            pred_pos2, pred_vel2, pred_vf2 = model(
                inputs2,
                current_num_phases=num_phases_tf,
                phase_densities=densities_tf,
                training=True,
                cd=0.5, cf=0.5
            )
            
            # 计算第二步损失
            loss2, pos_loss2, vf_loss2 = simple_loss_fn(
                pred_pos2, pos2_gt, pred_vf2, vf2_gt, current_p
            )
            
            # 总损失
            total_loss = 0.5 * loss1 + 0.5 * loss2
            
        # 计算梯度
        gradients = tape.gradient(total_loss, model.trainable_variables)
        
        # 梯度监控
        gradient_norms = []
        gradient_stats = []
        
        for i, (grad, var) in enumerate(zip(gradients, model.trainable_variables)):
            if grad is not None:
                grad_norm = tf.norm(grad)
                gradient_norms.append(grad_norm)
                gradient_stats.append({
                    'var_name': var.name,
                    'var_shape': var.shape,
                    'grad_norm': grad_norm.numpy(),
                    'grad_mean': tf.reduce_mean(tf.abs(grad)).numpy(),
                    'grad_max': tf.reduce_max(tf.abs(grad)).numpy(),
                    'has_nan': tf.reduce_any(tf.math.is_nan(grad)).numpy(),
                    'has_inf': tf.reduce_any(tf.math.is_inf(grad)).numpy()
                })
            else:
                print(f"Warning: No gradient for variable {var.name}")
        
        # 应用梯度
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return {
            'total_loss': total_loss.numpy(),
            'loss1': loss1.numpy(),
            'loss2': loss2.numpy(),
            'pos_loss1': pos_loss1.numpy(),
            'pos_loss2': pos_loss2.numpy(),
            'vf_loss1': vf_loss1 if isinstance(vf_loss1, float) else vf_loss1.numpy(),
            'vf_loss2': vf_loss2 if isinstance(vf_loss2, float) else vf_loss2.numpy(),
            'gradient_stats': gradient_stats,
            'total_grad_norm': tf.norm(gradient_norms).numpy() if gradient_norms else 0.0
        }

    # 执行多个训练步骤
    print("Running training steps...")
    try:
        for step in range(5):  # 运行5个训练步骤
            print(f"\n--- Training Step {step + 1} ---")
            
            # 执行训练步骤
            step_results = train_step()
            
            # 打印损失信息
            print(f"Total Loss: {step_results['total_loss']:.6f}")
            print(f"Step 1 - Pos Loss: {step_results['pos_loss1']:.6f}, VF Loss: {step_results['vf_loss1']:.6f}")
            print(f"Step 2 - Pos Loss: {step_results['pos_loss2']:.6f}, VF Loss: {step_results['vf_loss2']:.6f}")
            print(f"Total Gradient Norm: {step_results['total_grad_norm']:.6f}")
            
            # 检查梯度异常
            problematic_grads = [stat for stat in step_results['gradient_stats'] 
                               if stat['has_nan'] or stat['has_inf'] or stat['grad_norm'] > 100.0]
            
            if problematic_grads:
                print("Problematic gradients detected:")
                for stat in problematic_grads[:3]:  # 只显示前3个
                    print(f"  {stat['var_name']}: norm={stat['grad_norm']:.3f}, "
                          f"nan={stat['has_nan']}, inf={stat['has_inf']}")
            else:
                print("All gradients look healthy")
            
            # 显示一些代表性梯度统计
            if step == 0:  # 第一步显示详细信息
                print("\nDetailed Gradient Statistics:")
                for stat in step_results['gradient_stats'][:5]:  # 显示前5个变量
                    print(f"  {stat['var_name'][:50]}: "
                          f"norm={stat['grad_norm']:.4f}, "
                          f"mean={stat['grad_mean']:.4f}, "
                          f"max={stat['grad_max']:.4f}")
        
        print("\nSUCCESS! Training steps completed successfully!")
        print("The model can perform forward and backward passes without errors.")
        
    except Exception as e:
        print(f"\nFAILED! Training step error: {e}")
        import traceback
        traceback.print_exc()
        
        # 尝试CPU fallback
        print("\nTrying training on CPU...")
        try:
            with tf.device('/CPU:0'):
                step_results = train_step()
                print("CPU training step successful!")
        except Exception as cpu_e:
            print(f"CPU training also failed: {cpu_e}")

    # --- 8. 模型参数统计 ---
    print(f"\n--- Model Statistics ---")
    total_params = sum([tf.size(var).numpy() for var in model.trainable_variables])
    print(f"Total trainable parameters: {total_params:,}")
    
    # 按层统计参数
    layer_params = {}
    for var in model.trainable_variables:
        layer_name = var.name.split('/')[0] if '/' in var.name else 'other'
        if layer_name not in layer_params:
            layer_params[layer_name] = 0
        layer_params[layer_name] += tf.size(var).numpy()
    
    print("Parameters by component:")
    for layer_name, param_count in sorted(layer_params.items(), key=lambda x: x[1], reverse=True):
        print(f"  {layer_name}: {param_count:,}")