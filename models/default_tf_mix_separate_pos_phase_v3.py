#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
20251007
文件名: MultiPhaseParticleNetwork.py
描述: 
    本文件定义了一个用于数据驱动多相流体模拟的深度学习网络模型。
    该模型基于拉格朗日粒子法，并使用连续卷积作为核心的空间信息聚合算子。
    其设计目标是构建一个能够处理可变相数、多尺度物理属性，并保证
    数值稳定性和物理一致性的通用预测器。

核心特性:
1.  支持可变相数: 通过集成的深度集合编码器（DeepSetPhaseEncoder），
    模型可以处理包含不同数量相的模拟场景。
2.  鲁棒的特征处理: 采用基于物理先验的特征归一化策略（对数中心化与缩放），
    以应对密度等多尺度物理输入的挑战，防止训练过程中的数值不稳定问题。
3.  物理守恒的残差预测: 对体积分数（Volume Fraction）的预测采用
    在概率空间进行的残差更新机制，并通过两种可选方案（重新归一化或
    零和博弈）来严格强制质量守恒。
"""

import tensorflow as tf
import open3d.ml.tf as ml3d
import numpy as np
from debug_utils import debug_print
from models.deepset_encoder import DeepSetPhaseEncoder # 假设 DeepSetPhaseEncoder 在此路径下

import time

class MultiPhaseParticleNetwork(tf.keras.Model):
    """
    一个鲁棒且可泛化的数据驱动多相流体模拟网络。
    """
    def __init__(self,
                 # --- 模型核心参数 ---
                 max_num_phases: int = 5,
                 use_zero_sum_game: bool = False,
                 phase_feat_centralization: bool = True,
                 aggregation: str = 'mean',
                 # --- 网络结构参数 ---
                 kernel_size: list = [4, 4, 4],
                 shared_feature_channels: list = [64, 64, 64],
                 cd_cf_embedding_dim: int = 16,
                 # --- 物理与模拟参数 ---
                 particle_radius: float = 0.05,
                 radius_scale: float = 1.5,
                 timestep: float = 1 / 50,
                 gravity: tuple = (0, -9.81, 0),
                 # --- 其他卷积参数 ---
                 coordinate_mapping: str = 'ball_to_cube_volume_preserving',
                 interpolation: str = 'linear',
                 use_window: bool = True,
                 cd_cf_as_input: bool = True,
                 ) -> None:
        """
        初始化 MultiPhaseParticleNetwork 模型。

        Args:
            max_num_phases (int): 模型能够处理的最大相数。
            use_zero_sum_game (bool): 是否使用“零和博弈”机制来强制VF守恒。
                如果为 False，则使用“重新归一化”机制。
            phase_feat_centralization (bool): 是否对输入的多相特征进行零中心化处理。
            aggregation (str): DeepSet编码器的聚合方式 ('mean' 或 'sum')。
            kernel_size (list): 连续卷积核的大小。
            shared_feature_channels (list): 共享主干网络中各层的通道数。
            cd_cf_embedding_dim (int): 条件参数（cd/cf）的嵌入维度。
            particle_radius (float): 粒子的半径。
            radius_scale (float): 邻居搜索半径相对于粒子直径的比例因子。
            timestep (float): 模拟的时间步长。
            gravity (tuple): 重力加速度向量。
        """
        super().__init__(name=type(self).__name__)

        # --- 保存核心参数 ---
        self.max_num_phases = max_num_phases
        self.use_zero_sum_game = use_zero_sum_game
        self.phase_feat_centralization = phase_feat_centralization
        self.aggregation = aggregation
        
        self.shared_feature_channels = shared_feature_channels
        self.pos_output_channels = 3

        # --- [核心修改] 根据是否使用零和博弈，决定VF预测头的输出维度 ---
        if self.use_zero_sum_game:
            # 预测 N-1 个变化量
            self.vf_delta_output_dim = self.max_num_phases - 1
        else:
            # 预测 N 个变化量
            self.vf_delta_output_dim = self.max_num_phases
        
        # --- 保存其他参数 ---
        self.kernel_size = kernel_size
        self.radius_scale = radius_scale
        self.timestep = timestep
        self.gravity = tf.constant(gravity, dtype=tf.float32)
        # ... 其他参数 ...
        self.cd_cf_as_input = cd_cf_as_input
        self.cd_cf_embedding_dim = cd_cf_embedding_dim
        self.filter_extent = np.float32(self.radius_scale * 6 * particle_radius)
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.use_window = use_window

        # --- 初始化网络层 ---
        print(f"[Model INFO] Initializing model to handle up to {self.max_num_phases} phases.")
        print(f"[Model INFO] VF Conservation Strategy: {'Zero-Sum Game' if self.use_zero_sum_game else 'Re-normalization'}.")

        if self.cd_cf_as_input and self.cd_cf_embedding_dim > 0:
            self.cd_embedding_layer = tf.keras.layers.Dense(self.cd_cf_embedding_dim, activation='tanh', name='cd_embedding')
            self.cf_embedding_layer = tf.keras.layers.Dense(self.cd_cf_embedding_dim, activation='tanh', name='cf_embedding')

        self.phase_encoder = DeepSetPhaseEncoder(
            phi_dims=[64, 128], rho_dims=[128, 64], aggregation=self.aggregation)

        self._all_convs = []

        def window_poly6(r_sqr):
            r_sqr_clipped = tf.maximum(r_sqr, 0.0)
            return tf.clip_by_value((1 - r_sqr_clipped)**3, 0, 1)

        def Conv(name, activation=None, **kwargs):
            conv_fn = ml3d.layers.ContinuousConv
            window_fn = window_poly6 if self.use_window else None
            conv = conv_fn(name=name, kernel_size=self.kernel_size, activation=activation, align_corners=True,
                           interpolation=self.interpolation, coordinate_mapping=self.coordinate_mapping,
                           normalize=False, window_function=window_fn, radius_search_ignore_query_points=True, **kwargs)
            self._all_convs.append((name, conv))
            return conv

        # 主干网络层
        self.shared_conv0_fluid = Conv(name="shared_conv0_fluid", filters=self.shared_feature_channels[0])
        self.shared_conv0_obstacle = Conv(name="shared_conv0_obstacle", filters=self.shared_feature_channels[0])
        self.shared_dense0_fluid = tf.keras.layers.Dense(units=self.shared_feature_channels[0], name="shared_dense0_fluid")
        
        self.shared_convs, self.shared_denses = [], []
        for i, ch in enumerate(self.shared_feature_channels[1:], 1):
            self.shared_denses.append(tf.keras.layers.Dense(units=ch, name=f"shared_dense{i}"))
            self.shared_convs.append(Conv(name=f"shared_conv{i}", filters=ch))

        # 任务头
        self.pos_final_conv = Conv(name="pos_final_conv", filters=self.pos_output_channels)
        self.pos_final_dense = tf.keras.layers.Dense(units=self.pos_output_channels, name="pos_final_dense")
        
        if self.vf_delta_output_dim > 0:
            self.vf_final_conv = Conv(name="vf_final_conv", filters=self.vf_delta_output_dim)
            self.vf_final_dense = tf.keras.layers.Dense(units=self.vf_delta_output_dim, name="vf_final_dense")


    def call(self, inputs: tuple, current_num_phases: tf.Tensor = None, phase_densities: tf.Tensor = None, training: bool = False, **kwargs) -> tuple:
        """
        执行一次完整的前向传播。
        """
        # 如果未提供，使用默认值
        if current_num_phases is None:
            current_num_phases = tf.constant(self.max_num_phases, dtype=tf.int32)
        if phase_densities is None:
            phase_densities = tf.ones([self.max_num_phases], dtype=tf.float32) * 1000.0
        

        pos1, vel1, current_phase_fractions, box_pos, box_feats = inputs
        phase_densities = tf.convert_to_tensor(phase_densities, dtype=tf.float32)

        # 1. 物理积分
        pos2_integrated, vel2_integrated = self.integrate_pos_vel(pos1, vel1)

        # 2. 共享特征提取
        shared_features = self.compute_shared_features(
            pos2_integrated, vel2_integrated, current_phase_fractions, 
            current_num_phases, phase_densities, box_pos, box_feats, training=training, **kwargs)
        
        # 3. 位置预测
        pos_correction = self.compute_position_correction_from_shared(shared_features, pos2_integrated)
        pos_final, vel_final = self.compute_new_pos_vel(pos1, vel1, pos2_integrated, vel2_integrated, pos_correction)

        # 4. 体积分数预测
        next_phase_fractions_final = current_phase_fractions
        if self.max_num_phases > 1 and current_num_phases > 1:
            # 预测VF变化量
            predicted_delta_vf = self.compute_delta_vf_from_shared(shared_features, pos_final, current_num_phases)
            
            # 填充当前VF
            padding_size = self.max_num_phases - current_num_phases
            vf_current_padded = tf.pad(current_phase_fractions, [[0, 0], [0, padding_size]])
            
            # 应用残差更新和守恒机制
            vf_next_padded = self.compute_next_phase_fractions(vf_current_padded, predicted_delta_vf, current_num_phases)
            
            # 切片得到最终结果
            next_phase_fractions_final = vf_next_padded[:, :current_num_phases]
        
        return pos_final, vel_final, next_phase_fractions_final


    def compute_shared_features(self, pos, vel, phase_fractions, current_num_phases, phase_densities, box_pos, box_feats, training=False, **kwargs) -> tf.Tensor:
        """计算共享特征的核心函数。"""
        # --- 1. 特征预处理 ---
        densities_per_particle = tf.broadcast_to(phase_densities, tf.shape(phase_fractions))
        log_densities_per_particle = tf.math.log(densities_per_particle + 1e-8)

        if self.phase_feat_centralization:
            vf_scaled = (phase_fractions - 0.5) * 2.0
            LOG_DENSITY_CENTER, LOG_DENSITY_SCALE = 7.7, 1.5
            log_density_scaled = (log_densities_per_particle - LOG_DENSITY_CENTER) / LOG_DENSITY_SCALE
        else:
            vf_scaled = phase_fractions
            LOG_DENSITY_MIN, LOG_DENSITY_RANGE = 6.2146, 2.9957
            log_density_scaled = (log_densities_per_particle - LOG_DENSITY_MIN) / LOG_DENSITY_RANGE
        
        per_phase_features = tf.stack([vf_scaled, log_density_scaled], axis=-1)

        # --- 2. 结构化编码 ---
        padding_size = self.max_num_phases - current_num_phases
        padded_features = tf.pad(per_phase_features, [[0, 0], [0, padding_size], [0, 0]])
        
        mask_range = tf.range(self.max_num_phases)
        mask = mask_range < current_num_phases
        mask_per_particle = tf.broadcast_to(mask, [tf.shape(pos)[0], self.max_num_phases])

        particle_phase_embedding = self.phase_encoder((padded_features, mask_per_particle))

        # --- 3. 拼接最终特征 ---
        fluid_feats_list = [tf.ones_like(pos[:, 0:1]), vel, particle_phase_embedding]
        if self.cd_cf_as_input:
            cd_scalar = kwargs.get('cd_scalar', 0.5)
            cf_scalar = kwargs.get('cf_scalar', 0.5)
            cd_embed = self.cd_embedding_layer(tf.fill((tf.shape(pos)[0], 1), cd_scalar))
            cf_embed = self.cf_embedding_layer(tf.fill((tf.shape(pos)[0], 1), cf_scalar))
            fluid_feats_list.extend([cd_embed, cf_embed])
        fluid_feats = tf.concat(fluid_feats_list, axis=-1)
        
        # --- 4. 空间交互 (主干网络) ---
        filter_extent = tf.constant(self.filter_extent)
        shared_conv0_fluid = self.shared_conv0_fluid(fluid_feats, pos, pos, filter_extent)
        shared_dense0_fluid = self.shared_dense0_fluid(fluid_feats)
        shared_conv0_obstacle = self.shared_conv0_obstacle(box_feats, box_pos, pos, filter_extent)
        processed_feats = tf.concat([shared_conv0_obstacle, shared_conv0_fluid, shared_dense0_fluid], axis=-1)

        shared_ans_convs = [processed_feats]
        for conv, dense in zip(self.shared_convs, self.shared_denses):
            current_features = tf.keras.activations.relu(shared_ans_convs[-1])
            ans_conv = conv(current_features, pos, pos, filter_extent)
            ans_dense = dense(current_features)
            ans = ans_conv + ans_dense + (shared_ans_convs[-1] if ans_dense.shape[-1] == shared_ans_convs[-1].shape[-1] else 0)
            shared_ans_convs.append(ans)

        self.num_fluid_neighbors = ml3d.ops.reduce_subarrays_sum(tf.ones_like(self.shared_conv0_fluid.nns.neighbors_index, dtype=tf.float32), self.shared_conv0_fluid.nns.neighbors_row_splits)
        
        return tf.keras.activations.relu(shared_ans_convs[-1])


    def compute_next_phase_fractions(self, current_vf_padded: tf.Tensor, delta_vf_padded: tf.Tensor, current_num_phases: tf.Tensor) -> tf.Tensor:
        """
        在概率空间进行残差更新，并根据配置强制守恒。

        Args:
            current_vf_padded (tf.Tensor): 填充后的当前VF, 形状 [N, max_num_phases]。
            delta_vf_padded (tf.Tensor): 预测的VF变化量, 形状 [N, max_num_phases]。
            current_num_phases (tf.Tensor): 当前有效的相数。

        Returns:
            tf.Tensor: 下一时刻的VF（已填充），形状 [N, max_num_phases]。
        """
        # 对变化量进行缩放，作为一种学习率，增强稳定性
        vf_next_unnormalized = current_vf_padded + 0.1 * delta_vf_padded

        if self.use_zero_sum_game:
            # “零和博弈”下，delta_vf_padded 的有效区域和已经为0，
            # vf_next_unnormalized 的有效区域和理论上仍为1。
            # 只需裁剪以保证VF在[0,1]之间，并做一个轻微的再归一化处理数值误差。
            vf_next_clipped = tf.clip_by_value(vf_next_unnormalized, 0.0, 1.0)
            
            # 使用掩码确保只对有效相进行归一化
            mask_range = tf.range(self.max_num_phases, dtype=tf.int32)
            phase_mask = mask_range < current_num_phases
            phase_mask_float = tf.cast(tf.broadcast_to(phase_mask, tf.shape(vf_next_clipped)), dtype=tf.float32)
            
            vf_next_masked = vf_next_clipped * phase_mask_float
            sum_of_fractions = tf.reduce_sum(vf_next_masked, axis=1, keepdims=True)
            sum_of_fractions = tf.maximum(sum_of_fractions, 1e-8)
            return vf_next_masked / sum_of_fractions

        else:
            # “重新归一化”机制
            vf_next_non_negative = tf.keras.activations.relu(vf_next_unnormalized)
            
            mask_range = tf.range(self.max_num_phases, dtype=tf.int32)
            phase_mask = mask_range < current_num_phases
            phase_mask_float = tf.cast(tf.broadcast_to(phase_mask, tf.shape(vf_next_non_negative)), dtype=tf.float32)

            vf_next_masked = vf_next_non_negative * phase_mask_float
            sum_of_fractions = tf.reduce_sum(vf_next_masked, axis=1, keepdims=True)
            sum_of_fractions = tf.maximum(sum_of_fractions, 1e-8)
            return vf_next_masked / sum_of_fractions


    def compute_delta_vf_from_shared(self, shared_features: tf.Tensor, pos: tf.Tensor, current_num_phases: tf.Tensor) -> tf.Tensor:
        """
        从共享特征中计算体积分数的变化量 delta_vf。
        支持“零和博弈”和“重新归一化”两种模式。

        Args:
            shared_features (tf.Tensor): 共享主干网络的输出。
            pos (tf.Tensor): 粒子位置。
            current_num_phases (tf.Tensor): 当前场景的实际相数。

        Returns:
            tf.Tensor: 预测出的体积分数变化量，形状为 [N, max_num_phases]。
        """
        if self.vf_delta_output_dim <= 0:
            return tf.zeros(shape=(tf.shape(pos)[0], self.max_num_phases), dtype=pos.dtype)

        filter_extent = tf.constant(self.filter_extent)
        vf_conv = self.vf_final_conv(shared_features, pos, pos, filter_extent)
        vf_dense = self.vf_final_dense(shared_features)
        vf_delta_raw = vf_conv + vf_dense
        
        # 应用 tanh 激活来获得有界的变化量
        delta_vf_unstructured = tf.keras.activations.tanh(vf_delta_raw)

        if self.use_zero_sum_game:
            # --- “零和博弈”逻辑：预测 N-1 个，计算最后一个 ---
            # 1. 确定我们需要处理的维度。网络预测了 max_num_phases - 1 个 delta，
            #    但我们只关心与当前场景相关的 current_num_phases - 1 个。
            num_deltas_to_use = current_num_phases - 1
            # 从网络输出中只取前 num_deltas_to_use 个分量
            active_delta_N_minus_1 = delta_vf_unstructured[:, :num_deltas_to_use]
            
            # 2. 计算最后一个相的 delta，以保证在有效相上的和为零
            sum_of_deltas = tf.reduce_sum(active_delta_N_minus_1, axis=-1, keepdims=True)
            last_delta = -sum_of_deltas
            
            # 3. 拼接成完整的、和为零的、有效相的 delta 向量
            active_delta = tf.concat([active_delta_N_minus_1, last_delta], axis=-1)

            # 4. 将其填充回 max_num_phases 维度，以便后续计算
            padding_size = self.max_num_phases - current_num_phases
            return tf.pad(active_delta, [[0, 0], [0, padding_size]])
        else:
            # --- “重新归一化”逻辑：直接使用网络输出 ---
            return delta_vf_unstructured

    # --- 辅助方法 ---
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
        filter_extent = tf.constant(self.filter_extent)
        pos_conv = self.pos_final_conv(shared_features, pos, pos, filter_extent)
        pos_dense = self.pos_final_dense(shared_features)
        pos_output = pos_conv + pos_dense
        return (1.0 / 128.0) * pos_output

    def init(self, **kwargs):
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
        phase_fractions = np.zeros(shape=(1, init_num_phases), dtype=np.float32)
        phase_fractions[:, 0] = 1.0

        box = np.zeros(shape=(1, 3), dtype=np.float32)
        box_feats = np.zeros(shape=(1, 3), dtype=np.float32)
        
        # 创建匹配的虚拟物理属性
        init_densities = np.ones(shape=(init_num_phases,), dtype=np.float32) * 1000.0

        cd = np.float32(0.5)
        cf = np.float32(0.5)
        
        # 使用适配后的 call 方法签名进行调用
        _ = self.__call__((pos, vel, phase_fractions, box, box_feats),
                          current_num_phases=init_num_phases,
                          phase_densities=init_densities,
                          cd=cd, cf=cf)
        
        print(f"{self.name} initialized to handle up to {self.max_num_phases} phases (tested with {init_num_phases} phases).")
        print(f"Shared feature channels: {self.shared_feature_channels}")
        print(f"Position output channels: {self.pos_output_channels}")
        if self.max_num_phases > 1:
            print(f"VF output channels (max): {self.max_num_phases}")
        
        try:
            # 打印模型摘要
            self.summary()
        except Exception as e:
            print(f"Could not print model summary: {e}")