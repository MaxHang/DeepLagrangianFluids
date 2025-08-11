"""
灵活的多相流体粒子网络 - V0固定相数版

文件目的：
本文件定义了一个支持固定相数和动态物理属性的多相流预测器模型。
修改了V0版本，去掉了填充机制，使模型处理固定的相数。

版本特点 (V0-Fixed):
1.  **引入密度 (Density)**: 模型在每次调用时接受一个'phase_densities'列表，
    并根据粒子的体积分数计算'混合密度'，将其作为输入特征。

2.  **固定相数 (Fixed Number of Phases)**: 模型在初始化时确定固定的相数，
    避免了动态填充带来的复杂性和潜在问题。

3.  **简化的调用接口**: 去掉了复杂的填充和掩码机制，
    使模型更稳定且易于调试。

4.  **保持边界感知**: 确保固体边界的处理与原始模型一致。
"""

import tensorflow as tf
import open3d.ml.tf as ml3d
import numpy as np
from debug_utils import debug_print


class MultiPhaseParticleNetwork(tf.keras.Model):

    def __init__(self,
                 # --- 修改：使用固定相数 ---
                 num_phases=2, # 模型处理的固定相数
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

        self.num_phases = num_phases

        self.shared_feature_channels = [32, 64, 64]
        self.pos_output_channels = 3
        # 体积分数(VF)的输出通道数等于实际相数
        self.vf_output_channels = self.num_phases if num_phases > 1 else 0

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

        debug_print(f"Model initialized with fixed {self.num_phases} phases.")

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

    def call(self, inputs, phase_densities, training=False, fixed_radius_search_hash_table=None, cd=0.5, cf=0.5):
        """
        模型的前向传播 - 简化版本，不使用填充。
        Args:
            inputs (tuple): (pos1, vel1, current_phase_fractions, box_pos, box_feats)
            phase_densities (tf.Tensor or list): 当前相的密度列表/张量，长度必须等于num_phases。
        """
        pos1, vel1, current_phase_fractions, box_pos, box_feats = inputs
        
        phase_densities = tf.convert_to_tensor(phase_densities, dtype=tf.float32)
        
        # 验证输入维度
        if self.num_phases > 1:
            expected_vf_shape = tf.shape(current_phase_fractions)[-1]
            tf.debugging.assert_equal(expected_vf_shape, self.num_phases, 
                                    message=f"Phase fractions shape mismatch: expected {self.num_phases}")

        # 1. 积分位置和速度
        pos2_integrated, vel2_integrated = self.integrate_pos_vel(pos1, vel1)

        # 2. 计算共享特征
        shared_features = self.compute_shared_features(
            pos2_integrated, 
            vel2_integrated, 
            current_phase_fractions, 
            phase_densities, # 传入密度
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
        if self.num_phases > 1:
            # 预测下一时刻的体积分数logits
            next_vf_logits = self.compute_vf_logits_from_shared(shared_features, pos_final)
            
            # 计算下一时刻的体积分数
            next_phase_fractions_final = self.compute_next_phase_fractions(next_vf_logits)
        
        return pos_final, vel_final, next_phase_fractions_final

    def compute_shared_features(self, 
                                pos, 
                                vel, 
                                phase_fractions, 
                                phase_densities, 
                                box_pos, 
                                box_feats,
                                fixed_radius_search_hash_table=None, 
                                cd_scalar=0.5, 
                                cf_scalar=0.5):
        filter_extent_tensor = tf.constant(self.filter_extent, dtype=tf.float32)

        # --- 不使用填充，直接使用原始相分数 ---

        # 方案1：使用相对密度比值（推荐用于泛化）
    # 计算密度比值：每相密度相对于最轻相的比值
        min_density = tf.reduce_min(phase_densities)
        relative_phase_densities = phase_densities / min_density  # 范围: [1.0, density_ratio]
        # 计算混合密度
        mixture_density = tf.reduce_sum(phase_fractions * relative_phase_densities, axis=-1, keepdims=True)

        debug_print(f"mixture_density shape: {mixture_density.shape}")

        fluid_feats_list = [
            tf.ones_like(pos[:, 0:1]),   # 1. 存在特征
            vel,                         # 2. 当前速度
            mixture_density,             # 3. 混合密度
        ]

        # 只有在多相情况下才添加相分数特征
        if self.num_phases > 1 and phase_fractions is not None:
            fluid_feats_list.append(phase_fractions)  # 4. 各相体积分数

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

    def compute_next_phase_fractions(self, network_vf_logits):
        """
        简化的相分数计算，不使用掩码。
        Args:
            network_vf_logits (tf.Tensor): [N, num_phases] 来自网络的原始输出
        """
        if self.num_phases <= 1:
            return None
            
        # 直接应用softmax，不需要掩码
        next_fractions = tf.nn.softmax(network_vf_logits, axis=-1)
        
        return next_fractions
    
    # --- 辅助方法（与原始代码一致） ---

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
        """
        # 创建符合固定相数的虚拟输入数据
        pos = np.zeros(shape=(1, 3), dtype=np.float32)
        vel = np.zeros(shape=(1, 3), dtype=np.float32)
        
        # 体积分数张量的维度由 num_phases 决定
        if self.num_phases > 1:
            phase_fractions = np.zeros(shape=(1, self.num_phases), dtype=np.float32)
            phase_fractions[:, 0] = 1.0
        else:
            phase_fractions = None

        box = np.zeros(shape=(1, 3), dtype=np.float32)
        box_feats = np.zeros(shape=(1, 3), dtype=np.float32)
        
        # 创建匹配的虚拟物理属性
        densities = np.ones(shape=(self.num_phases,), dtype=np.float32) * 1000.0

        cd = np.float32(0.5)
        cf = np.float32(0.5)
        
        # 使用简化的 call 方法签名进行调用
        _ = self.__call__((pos, vel, phase_fractions, box, box_feats),
                          phase_densities=densities,
                          cd=cd, cf=cf)
        
        print(f"{self.name} initialized with fixed {self.num_phases} phases.")
        print(f"Shared feature channels: {self.shared_feature_channels}")
        print(f"Position output channels: {self.pos_output_channels}")
        if self.num_phases > 1:
            print(f"VF output channels: {self.vf_output_channels}")
        
        try:
            # 打印模型摘要
            self.summary()
        except Exception as e:
            print(f"Could not print model summary: {e}")


# --- 使用示例 ---
#
# # 1. 初始化模型时，指定固定相数
# model = MultiPhaseParticleNetwork(num_phases=2)
#
# # 2. 准备2相流的训练样本
# densities_2 = [1000.0, 500.0] # 水和油
# # 假设 inputs_2_phase 是一个包含 (pos, vel, vf, box_pos, box_feats) 的元组
# # 其中 vf 的形状是 [num_particles, 2]
# # pred_pos, pred_vel, pred_vf = model(inputs_2_phase, phase_densities=densities_2)
# # pred_vf 的形状将是 [num_particles, 2]