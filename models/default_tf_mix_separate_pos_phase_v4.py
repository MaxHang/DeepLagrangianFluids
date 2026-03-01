#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
创建日期: 2025-10-20
文件名: default_tf_mix_separate_pos_phase_v4.py
版本: V4

描述: 
    本文件定义了一个用于数据驱动多相流体模拟的深度学习网络模型（第4版）。
    该模型基于拉格朗日粒子法，并使用连续卷积作为核心的空间信息聚合算子。
    
V4 版本的核心改进:
    1. 完全移除 padding 依赖：
       - 输入端使用 DeepSet v2，支持动态相数输入（无需 padding 到 max_num_phases）
       - 输出端使用动态 VF 预测头，根据实际相数创建对应维度的网络层
    
    2. 更精确的质量守恒：
       - 消除了 padding 引入的数值误差和梯度噪声
       - 零和博弈机制在真实相数上严格满足 Σ(delta_vf) = 0
       - 重新归一化机制只对有效相进行计算
    
    3. 提升计算效率：
       - 避免在填充维度上的冗余计算
       - 动态层缓存机制减少运行时开销
    
    4. 增强数值稳定性：
       - 特征归一化策略保持不变（对数中心化与缩放）
       - 梯度传播更加干净，无 padding 相关的噪声

核心设计原则:
    - 真正的多相流模型应该对不同相数使用相应维度的子网络
    - 物理守恒约束应该在实际相数的空间中严格满足
    - 网络结构应该与物理问题的本质匹配，而非通过 padding 来适配
"""

import tensorflow as tf
import open3d.ml.tf as ml3d
import numpy as np
from typing import Tuple, Dict, List, Optional
from debug_utils import debug_print
from models.deepset_encoder_v2 import DeepSetPhaseEncoder

import time

class MultiPhaseParticleNetwork(tf.keras.Model):
    """
    鲁棒且可泛化的数据驱动多相流体模拟网络（V4版本）。
    
    该网络支持：
    - 动态相数处理（2 至 max_num_phases 相）
    - 严格的质量守恒（通过零和博弈或重新归一化）
    - 多尺度物理属性的鲁棒特征编码
    - 端到端的粒子位置和相分布联合预测
    """
    
    def __init__(self,
                 # --- 模型核心参数 ---
                 max_num_phases: int = 5,  # V4: 可选参数，用于预创建预测头
                 use_zero_sum_game: bool = False,
                 phase_feat_centralization: bool = True,
                 aggregation: str = 'mean',
                 # --- 网络结构参数 ---
                 kernel_size: List[int] = [4, 4, 4],
                 shared_feature_channels: List[int] = [64, 64, 64],
                 cd_cf_embedding_dim: int = 16,
                 # --- 物理与模拟参数 ---
                 particle_radius: float = 0.05,
                 radius_scale: float = 1.5,
                 timestep: float = 1 / 50,
                 gravity: Tuple[float, float, float] = (0, -9.81, 0),
                 # --- 其他卷积参数 ---
                 coordinate_mapping: str = 'ball_to_cube_volume_preserving',
                 interpolation: str = 'linear',
                 use_window: bool = True,
                 cd_cf_as_input: bool = True,
                 ) -> None:
        """
        初始化 MultiPhaseParticleNetwork 模型（V4版本）。

        Args:
            max_num_phases: 模型能够处理的最大相数。网络会为 [2, max_num_phases] 范围内的
                每个相数预先创建对应的 VF 预测头，避免运行时开销。
            use_zero_sum_game: 是否使用"零和博弈"机制来强制 VF 守恒。
                - True: 预测 N-1 个相的变化量，最后一相通过 -Σ(delta) 计算，严格保证和为0
                - False: 预测 N 个相的变化量，通过重新归一化保证和为1
            phase_feat_centralization: 是否对输入的多相特征进行零中心化处理。
                - True: VF ∈ [-1, 1], log(density) 零均值归一化（推荐用于训练稳定性）
                - False: VF ∈ [0, 1], log(density) 最小-最大归一化
            aggregation: DeepSet 编码器的聚合方式。
                - 'mean': 对有效相的特征求平均（推荐，对相数变化更鲁棒）
                - 'sum': 对有效相的特征求和
            kernel_size: 连续卷积核的大小，格式为 [k_x, k_y, k_z]。
            shared_feature_channels: 共享主干网络中各层的通道数。
                例如 [64, 64, 64] 表示3层，每层64个通道。
            cd_cf_embedding_dim: 条件参数（漂移系数 cd / 扩散系数 cf）的嵌入维度。
                设为0则不使用条件参数。
            particle_radius: 粒子的半径（米）。
            radius_scale: 邻居搜索半径相对于粒子直径的比例因子。
                实际搜索半径 = radius_scale * 2 * particle_radius。
            timestep: 模拟的时间步长（秒）。
            gravity: 重力加速度向量 (g_x, g_y, g_z)，单位 m/s²。
            coordinate_mapping: 连续卷积的坐标映射方式。
            interpolation: 连续卷积的插值方式（'linear' 或 'nearest'）。
            use_window: 是否在连续卷积中使用窗口函数（Poly6核）。
            cd_cf_as_input: 是否将 cd/cf 作为条件输入。
        """
        super().__init__(name=type(self).__name__)

        # ========== 保存核心参数 ==========
        self.max_num_phases = max_num_phases
        self.use_zero_sum_game = use_zero_sum_game
        self.phase_feat_centralization = phase_feat_centralization
        self.aggregation = aggregation
        
        self.shared_feature_channels = shared_feature_channels
        self.pos_output_channels = 3  # (x, y, z) 位置修正
        
        # ========== 保存其他参数 ==========
        self.kernel_size = kernel_size
        self.radius_scale = radius_scale
        self.timestep = timestep
        self.gravity = tf.constant(gravity, dtype=tf.float32)
        self.cd_cf_as_input = cd_cf_as_input
        self.cd_cf_embedding_dim = cd_cf_embedding_dim
        self.filter_extent = np.float32(self.radius_scale * 6 * particle_radius)
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.use_window = use_window

        # ========== 打印初始化信息 ==========
        print(f"\n{'='*80}")
        print(f"[Model V4] Initializing MultiPhaseParticleNetwork")
        print(f"{'='*80}")
        print(f"  Max phases supported    : {self.max_num_phases}")
        print(f"  VF conservation strategy: {'Zero-Sum Game' if self.use_zero_sum_game else 'Re-normalization'}")
        print(f"  Phase feature encoding  : {'Centralized' if self.phase_feat_centralization else 'MinMax Scaled'}")
        print(f"  DeepSet aggregation     : {self.aggregation}")
        print(f"  Shared feature channels : {self.shared_feature_channels}")
        print(f"  Filter extent           : {self.filter_extent:.4f} m")
        print(f"{'='*80}\n")

        # ========== 条件嵌入层（cd/cf）==========
        if self.cd_cf_as_input and self.cd_cf_embedding_dim > 0:
            self.cd_embedding_layer = tf.keras.layers.Dense(
                self.cd_cf_embedding_dim, 
                activation='tanh', 
                name='cd_embedding'
            )
            self.cf_embedding_layer = tf.keras.layers.Dense(
                self.cd_cf_embedding_dim, 
                activation='tanh', 
                name='cf_embedding'
            )

        # ========== DeepSet 相编码器（V2版本）==========
        # 输入: [N, current_num_phases, 2]  (无需 padding)
        # 输出: [N, 64]  (固定维度的相嵌入)
        self.phase_encoder = DeepSetPhaseEncoder(
            phi_dims=[64, 128],      # φ 网络: 2 -> 64 -> 128
            rho_dims=[128, 64],      # ρ 网络: 128 -> 128 -> 64
            aggregation=self.aggregation
        )

        # ========== 卷积层管理 ==========
        self._all_convs = []  # 用于跟踪所有卷积层（调试用）

        def window_poly6(r_sqr: tf.Tensor) -> tf.Tensor:
            """Poly6 窗口函数，用于平滑邻居权重。"""
            r_sqr_clipped = tf.maximum(r_sqr, 0.0)
            return tf.clip_by_value((1 - r_sqr_clipped)**3, 0, 1)

        def Conv(name: str, filters: int, activation=None, **kwargs) -> ml3d.layers.ContinuousConv:
            """卷积层构建辅助函数。"""
            window_fn = window_poly6 if self.use_window else None
            conv = ml3d.layers.ContinuousConv(
                name=name, 
                filters=filters,
                kernel_size=self.kernel_size, 
                activation=activation, 
                align_corners=True,
                interpolation=self.interpolation, 
                coordinate_mapping=self.coordinate_mapping,
                normalize=False, 
                window_function=window_fn, 
                radius_search_ignore_query_points=True, 
                **kwargs
            )
            self._all_convs.append((name, conv))
            return conv

        # ========== 共享主干网络 ==========
        # 第0层: 初始特征提取
        # 输入维度: [N, feat_dim] 其中 feat_dim = 1 + 3 + 64 + (cd_embed) + (cf_embed)
        self.shared_conv0_fluid = Conv(
            name="shared_conv0_fluid", 
            filters=self.shared_feature_channels[0]
        )
        self.shared_conv0_obstacle = Conv(
            name="shared_conv0_obstacle", 
            filters=self.shared_feature_channels[0]
        )
        self.shared_dense0_fluid = tf.keras.layers.Dense(
            units=self.shared_feature_channels[0], 
            name="shared_dense0_fluid"
        )
        
        # 后续层: 特征细化
        # 使用残差连接增强梯度流
        self.shared_convs: List[ml3d.layers.ContinuousConv] = []
        self.shared_denses: List[tf.keras.layers.Dense] = []
        for i, ch in enumerate(self.shared_feature_channels[1:], 1):
            self.shared_denses.append(
                tf.keras.layers.Dense(units=ch, name=f"shared_dense{i}")
            )
            self.shared_convs.append(
                Conv(name=f"shared_conv{i}", filters=ch)
            )

        # ========== 位置预测头（固定维度）==========
        # 输入: [N, shared_feature_channels[-1]]
        # 输出: [N, 3] (x, y, z 位置修正)
        self.pos_final_conv = Conv(
            name="pos_final_conv", 
            filters=self.pos_output_channels
        )
        self.pos_final_dense = tf.keras.layers.Dense(
            units=self.pos_output_channels, 
            name="pos_final_dense"
        )
        
        # ========== 动态 VF 预测头（V4 核心改进）==========
        # 为每个可能的相数预先创建对应的预测层
        # 相数范围: [2, max_num_phases]
        self.vf_heads: Dict[int, Dict[str, tf.keras.layers.Layer]] = {}
        
        if self.max_num_phases > 1:
            print(f"[Model V4] Creating dynamic VF prediction heads...")
            for num_phases in range(2, self.max_num_phases + 1):
                # 根据守恒策略确定输出维度
                if self.use_zero_sum_game:
                    output_dim = num_phases - 1  # 预测 N-1 个，最后一个通过 -Σ 计算
                else:
                    output_dim = num_phases  # 预测所有 N 个
                
                # 创建该相数对应的卷积和全连接层
                self.vf_heads[num_phases] = {
                    'conv': Conv(
                        name=f"vf_conv_p{num_phases}", 
                        filters=output_dim
                    ),
                    'dense': tf.keras.layers.Dense(
                        units=output_dim, 
                        name=f"vf_dense_p{num_phases}"
                    ),
                    'output_dim': output_dim
                }
                print(f"    Phase {num_phases}: output_dim = {output_dim}")
            print(f"[Model V4] Dynamic VF heads created for {len(self.vf_heads)} configurations.\n")

    def call(self, 
             inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], 
             current_num_phases: Optional[tf.Tensor] = None, 
             phase_densities: Optional[tf.Tensor] = None, 
             training: bool = False, 
             **kwargs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        执行一次完整的前向传播。

        Args:
            inputs: 包含5个张量的元组
                - pos1: 当前时刻粒子位置, shape [N, 3]
                - vel1: 当前时刻粒子速度, shape [N, 3]
                - current_phase_fractions: 当前时刻体积分数, shape [N, current_num_phases]
                - box_pos: 障碍物粒子位置, shape [M, 3]
                - box_feats: 障碍物粒子特征, shape [M, feature_dim]
            current_num_phases: 当前场景的实际相数（标量张量）。
                如果为 None，默认使用 max_num_phases。
            phase_densities: 各相的密度, shape [current_num_phases] 或 [max_num_phases]。
                如果为 None，默认所有相密度为 1000.0 kg/m³。
            training: 是否处于训练模式。
            **kwargs: 其他参数，如 cd, cf。

        Returns:
            三元组 (pos_final, vel_final, next_phase_fractions_final)
                - pos_final: 下一时刻粒子位置, shape [N, 3]
                - vel_final: 下一时刻粒子速度, shape [N, 3]
                - next_phase_fractions_final: 下一时刻体积分数, shape [N, current_num_phases]
        
        计算流程:
            1. 物理积分: 使用半隐式欧拉法进行初步位置/速度更新
            2. 特征提取: 通过 DeepSet 和共享主干网络提取多尺度特征
            3. 位置预测: 基于共享特征计算位置修正量
            4. VF 预测: 基于共享特征和动态预测头计算相分布变化
        """
        # ========== 参数默认值处理 ==========
        if current_num_phases is None:
            current_num_phases = tf.constant(self.max_num_phases, dtype=tf.int32)
        if phase_densities is None:
            phase_densities = tf.ones([self.max_num_phases], dtype=tf.float32) * 1000.0
        
        # 解包输入
        pos1, vel1, current_phase_fractions, box_pos, box_feats = inputs
        phase_densities = tf.convert_to_tensor(phase_densities, dtype=tf.float32)

        # ========== 1. 物理积分（半隐式欧拉法）==========
        # pos2 = pos1 + dt * (vel1 + vel2) / 2
        # vel2 = vel1 + dt * gravity
        # 输入: pos1 [N, 3], vel1 [N, 3]
        # 输出: pos2_integrated [N, 3], vel2_integrated [N, 3]
        pos2_integrated, vel2_integrated = self.integrate_pos_vel(pos1, vel1)

        # ========== 2. 共享特征提取 ==========
        # 输入: pos [N, 3], vel [N, 3], phase_fractions [N, current_num_phases], ...
        # 输出: shared_features [N, shared_feature_channels[-1]]
        shared_features = self.compute_shared_features(
            pos2_integrated, vel2_integrated, current_phase_fractions, 
            current_num_phases, phase_densities, box_pos, box_feats, 
            training=training, **kwargs
        )
        
        # ========== 3. 位置预测 ==========
        # 输入: shared_features [N, C], pos2_integrated [N, 3]
        # 输出: pos_correction [N, 3]
        pos_correction = self.compute_position_correction_from_shared(
            shared_features, pos2_integrated
        )
        # 最终位置和速度
        # pos_final [N, 3], vel_final [N, 3]
        pos_final, vel_final = self.compute_new_pos_vel(
            pos1, vel1, pos2_integrated, vel2_integrated, pos_correction
        )

        # ========== 4. 体积分数预测（V4 改进：无 padding）==========
        next_phase_fractions_final = current_phase_fractions
        
        if current_num_phases > 1:
            # 使用动态预测头直接预测实际维度的 delta_vf
            # 输入: shared_features [N, C], pos_final [N, 3], current_num_phases (scalar)
            # 输出: predicted_delta_vf [N, current_num_phases]
            predicted_delta_vf = self.compute_delta_vf_from_shared(
                shared_features, pos_final, current_num_phases
            )
            
            # 应用残差更新和守恒机制（直接在实际维度上操作，无需 padding）
            # 输入: current_phase_fractions [N, current_num_phases], 
            #       predicted_delta_vf [N, current_num_phases]
            # 输出: next_phase_fractions_final [N, current_num_phases]
            next_phase_fractions_final = self.compute_next_phase_fractions(
                current_phase_fractions, 
                predicted_delta_vf, 
                current_num_phases
            )
        
        return pos_final, vel_final, next_phase_fractions_final

    def compute_shared_features(self, 
                                pos: tf.Tensor, 
                                vel: tf.Tensor, 
                                phase_fractions: tf.Tensor, 
                                current_num_phases: tf.Tensor, 
                                phase_densities: tf.Tensor, 
                                box_pos: tf.Tensor, 
                                box_feats: tf.Tensor, 
                                training: bool = False, 
                                **kwargs) -> tf.Tensor:
        """
        计算共享特征的核心函数（V4改进：使用 DeepSet v2，无需 padding）。

        Args:
            pos: 粒子位置, shape [N, 3]
            vel: 粒子速度, shape [N, 3]
            phase_fractions: 当前体积分数, shape [N, current_num_phases]
            current_num_phases: 当前场景实际相数（标量）
            phase_densities: 各相密度, shape [current_num_phases] 或更长
            box_pos: 障碍物位置, shape [M, 3]
            box_feats: 障碍物特征, shape [M, F]
            training: 训练模式标志
            **kwargs: 额外参数（cd, cf 等）

        Returns:
            共享特征张量, shape [N, shared_feature_channels[-1]]
        
        处理流程:
            1. 特征归一化: VF 和 log(density) 的标准化处理
            2. DeepSet 编码: 将可变数量的相特征编码为固定维度
            3. 特征拼接: 组合物理特征和条件参数
            4. 空间交互: 通过连续卷积聚合邻居信息
        """
        # ========== 1. 多相特征预处理（无需 padding）==========
        # 只提取当前场景所需的相密度
        # current_densities: [current_num_phases]
        current_densities = phase_densities[:current_num_phases]
        
        # 广播到每个粒子的每个相
        # densities_per_particle: [N, current_num_phases]
        densities_per_particle = tf.broadcast_to(
            current_densities, 
            tf.shape(phase_fractions)
        )
        
        # 对数变换（处理多尺度密度范围）
        # log_densities_per_particle: [N, current_num_phases]
        log_densities_per_particle = tf.math.log(densities_per_particle + 1e-8)

        # 特征归一化（两种策略）
        if self.phase_feat_centralization:
            # 策略1: 零中心化（推荐用于训练）
            # VF: [0, 1] -> [-1, 1]
            vf_scaled = (phase_fractions - 0.5) * 2.0
            
            # log(density): 基于训练集统计的中心化
            # 假设密度范围 [500, 3000] kg/m³ -> log(density) ∈ [6.2, 8.0]
            # 中心值 ≈ 7.7, 标准差 ≈ 1.5
            LOG_DENSITY_CENTER, LOG_DENSITY_SCALE = 7.7, 1.5
            log_density_scaled = (log_densities_per_particle - LOG_DENSITY_CENTER) / LOG_DENSITY_SCALE
        else:
            # 策略2: 最小-最大归一化
            vf_scaled = phase_fractions
            
            # log(density): [6.2146, 9.2103] -> [0, 1]
            LOG_DENSITY_MIN, LOG_DENSITY_RANGE = 6.2146, 2.9957
            log_density_scaled = (log_densities_per_particle - LOG_DENSITY_MIN) / LOG_DENSITY_RANGE
        
        # 堆叠为每个相的特征向量
        # per_phase_features: [N, current_num_phases, 2]
        #   第3维: [scaled_vf, scaled_log_density]
        per_phase_features = tf.stack([vf_scaled, log_density_scaled], axis=-1)

        # ========== 2. DeepSet 相编码（V4改进：无 padding）==========
        # 创建全为 True 的 mask（因为不再需要 padding）
        # mask_per_particle: [N, current_num_phases], all True
        mask_per_particle = tf.ones(
            [tf.shape(pos)[0], current_num_phases], 
            dtype=tf.bool
        )

        # DeepSet 编码器
        # 输入: per_phase_features [N, current_num_phases, 2], mask [N, current_num_phases]
        # 输出: particle_phase_embedding [N, 64]
        particle_phase_embedding = self.phase_encoder((per_phase_features, mask_per_particle))

        # ========== 3. 特征拼接 ==========
        # 基础特征: [常数项, 速度, 相嵌入]
        # fluid_feats: [N, 1+3+64] = [N, 68]
        fluid_feats_list = [
            tf.ones_like(pos[:, 0:1]),      # [N, 1] 常数偏置项
            vel,                             # [N, 3] 速度
            particle_phase_embedding         # [N, 64] 相嵌入
        ]
        
        # 可选: 添加条件参数（漂移/扩散系数）
        if self.cd_cf_as_input:
            cd_scalar = kwargs.get('cd', 0.5)  # 默认值 0.5
            cf_scalar = kwargs.get('cf', 0.5)
            
            # cd_embed, cf_embed: [N, cd_cf_embedding_dim]
            cd_embed = self.cd_embedding_layer(
                tf.fill((tf.shape(pos)[0], 1), cd_scalar)
            )
            cf_embed = self.cf_embedding_layer(
                tf.fill((tf.shape(pos)[0], 1), cf_scalar)
            )
            fluid_feats_list.extend([cd_embed, cf_embed])
        
        # 最终特征拼接
        # fluid_feats: [N, total_feat_dim]
        fluid_feats = tf.concat(fluid_feats_list, axis=-1)
        
        # ========== 4. 空间交互（连续卷积主干网络）==========
        filter_extent = tf.constant(self.filter_extent)
        
        # 第0层: 初始特征提取
        # shared_conv0_fluid: [N, C0] - 流体粒子自身交互
        shared_conv0_fluid = self.shared_conv0_fluid(
            fluid_feats, pos, pos, filter_extent
        )
        # shared_dense0_fluid: [N, C0] - 逐粒子特征变换
        shared_dense0_fluid = self.shared_dense0_fluid(fluid_feats)
        
        # shared_conv0_obstacle: [N, C0] - 流体-障碍物交互
        shared_conv0_obstacle = self.shared_conv0_obstacle(
            box_feats, box_pos, pos, filter_extent
        )
        
        # 组合所有初始特征
        # processed_feats: [N, 3*C0]
        processed_feats = tf.concat([
            shared_conv0_obstacle, 
            shared_conv0_fluid, 
            shared_dense0_fluid
        ], axis=-1)

        # 后续层: 带残差连接的特征细化
        shared_ans_convs = [processed_feats]
        for conv, dense in zip(self.shared_convs, self.shared_denses):
            current_features = tf.keras.activations.relu(shared_ans_convs[-1])
            
            # ans_conv: [N, Ci]
            ans_conv = conv(current_features, pos, pos, filter_extent)
            # ans_dense: [N, Ci]
            ans_dense = dense(current_features)
            
            # 残差连接（如果维度匹配）
            residual = shared_ans_convs[-1] if ans_dense.shape[-1] == shared_ans_convs[-1].shape[-1] else 0
            ans = ans_conv + ans_dense + residual
            
            shared_ans_convs.append(ans)

        # 保存邻居数量（用于调试和可视化）
        self.num_fluid_neighbors = ml3d.ops.reduce_subarrays_sum(
            tf.ones_like(self.shared_conv0_fluid.nns.neighbors_index, dtype=tf.float32), 
            self.shared_conv0_fluid.nns.neighbors_row_splits
        )
        
        # 返回最终共享特征
        # shared_features: [N, shared_feature_channels[-1]]
        return tf.keras.activations.relu(shared_ans_convs[-1])
    
    def compute_delta_vf_from_shared(self, 
                                     shared_features: tf.Tensor, 
                                     pos: tf.Tensor, 
                                     current_num_phases: tf.Tensor) -> tf.Tensor:
        """
        从共享特征中计算体积分数的变化量（V4改进：动态预测头，无 padding）。

        Args:
            shared_features: 共享主干网络输出, shape [N, C]
            pos: 粒子位置, shape [N, 3]
            current_num_phases: 当前场景实际相数（标量张量）

        Returns:
            预测的体积分数变化量, shape [N, current_num_phases]
        
        工作原理:
            - 根据 current_num_phases 选择对应的预测头
            - 零和博弈模式: 预测 N-1 个相，最后一相通过 -Σ(delta) 计算
            - 重新归一化模式: 预测所有 N 个相

        分支映射逻辑：
            current_num_phases=2 → branch_index=0 → self.vf_heads[2]
            current_num_phases=3 → branch_index=1 → self.vf_heads[3]
            current_num_phases=4 → branch_index=2 → self.vf_heads[4]
        """        
        # ========== 使用 tf.switch_case 动态选择预测头 ==========
        filter_extent = tf.constant(self.filter_extent)
        
        # 单相场景无需预测 VF，直接返回零张量
        # 注意：这里不能用 if/else，因为会触发 AutoGraph
        # 我们通过 tf.cond 来处理
        def compute_vf_changes():
            """计算多相场景的 VF 变化"""
            def create_branch_fn(num_phases):
                """为每个相数创建一个分支函数"""

                def branch_fn():
                    vf_head = self.vf_heads[num_phases]
                    vf_conv = vf_head['conv']
                    vf_dense = vf_head['dense']
                    
                    # ========== 前向传播 ==========
                    # vf_conv_out: [N, output_dim]
                    vf_conv_out = vf_conv(shared_features, pos, pos, filter_extent)
                    # vf_dense_out: [N, output_dim]
                    vf_dense_out = vf_dense(shared_features)
                    # 组合卷积和全连接输出
                    vf_delta_raw = vf_conv_out + vf_dense_out
                    # 应用 tanh 激活，将变化量限制在 [-1, 1]
                    # delta_vf_partial: [N, output_dim]
                    delta_vf_partial = tf.keras.activations.tanh(vf_delta_raw)
                    
                    # ========== 根据守恒策略处理输出 ==========
                    if self.use_zero_sum_game:
                        # 零和博弈: output_dim = current_num_phases - 1
                        # 计算最后一相的 delta，保证 Σ(delta) = 0
                        sum_of_deltas = tf.reduce_sum(delta_vf_partial, axis=-1, keepdims=True)
                        last_delta = -sum_of_deltas
                        # 拼接完整的 delta 向量
                        # delta_vf_complete: [N, current_num_phases]
                        delta_vf_complete = tf.concat([delta_vf_partial, last_delta], axis=-1)
                        return delta_vf_complete
                    else:
                        # 重新归一化: 直接返回
                        return delta_vf_partial
                
                return branch_fn
            
            # 构建分支字典：{phase_index: branch_function}
            # phase_index = num_phases - 2 (因为从2开始)
            branch_fns = {}
            for num_phases in range(2, self.max_num_phases + 1):
                branch_index = num_phases - 2  # 2->0, 3->1, 4->2, ...
                branch_fns[branch_index] = create_branch_fn(num_phases)
            
            # 使用 tf.switch_case 根据 current_num_phases 选择分支
            # current_num_phases=2 -> branch_index=0
            # current_num_phases=3 -> branch_index=1
            branch_index = current_num_phases - 2
            
            predicted_delta_vf = tf.switch_case(
                branch_index,
                branch_fns,
                default=lambda: tf.zeros(
                    shape=(tf.shape(pos)[0], current_num_phases), 
                    dtype=pos.dtype
                )
            )
            
            return predicted_delta_vf
        
        def return_zeros():
            """单相场景返回零张量"""
            return tf.zeros(shape=(tf.shape(pos)[0], current_num_phases), dtype=pos.dtype)
    
        # 使用 tf.cond 根据相数决定是否计算 VF 变化
        result = tf.cond(
            current_num_phases > 1,
            true_fn=compute_vf_changes,
            false_fn=return_zeros
        )
        
        return result


    def compute_next_phase_fractions(self, 
                                     current_vf: tf.Tensor, 
                                     delta_vf: tf.Tensor, 
                                     current_num_phases: tf.Tensor) -> tf.Tensor:
        """
        在概率空间进行残差更新，并强制守恒（V4改进：直接在实际维度操作）。

        Args:
            current_vf: 当前体积分数, shape [N, current_num_phases]
            delta_vf: 预测的变化量, shape [N, current_num_phases]
            current_num_phases: 当前相数（标量）

        Returns:
            下一时刻的体积分数, shape [N, current_num_phases]
            满足约束: Σ(vf) = 1, vf ∈ [0, 1]
        
        守恒机制:
            - 零和博弈: delta_vf 已满足 Σ(delta) = 0，通过裁剪+归一化保证物理约束
            - 重新归一化: 通过 ReLU + 归一化强制满足约束
        """
        # ========== 残差更新 ==========
        # 学习率设为 0.1 以增强数值稳定性
        # vf_next_unnormalized: [N, current_num_phases]
        vf_next_unnormalized = current_vf + 0.1 * delta_vf

        if self.use_zero_sum_game:
            # ========== 零和博弈模式 ==========
            # 理论上 Σ(vf_next_unnormalized) = Σ(current_vf) = 1
            # 但由于数值误差，需要进行裁剪和归一化
            
            # 1. 裁剪到 [0, 1]
            vf_next_clipped = tf.clip_by_value(vf_next_unnormalized, 0.0, 1.0)
            
            # 2. 重新归一化以消除数值误差
            sum_of_fractions = tf.reduce_sum(vf_next_clipped, axis=1, keepdims=True)
            sum_of_fractions = tf.maximum(sum_of_fractions, 1e-8)  # 避免除以零
            
            # vf_next: [N, current_num_phases], Σ(vf_next) = 1
            vf_next = vf_next_clipped / sum_of_fractions
            
            return vf_next
        else:
            # ========== 重新归一化模式 ==========
            # 1. 使用 ReLU 保证非负性
            vf_next_non_negative = tf.keras.activations.relu(vf_next_unnormalized)
            
            # 2. 归一化到概率空间
            sum_of_fractions = tf.reduce_sum(vf_next_non_negative, axis=1, keepdims=True)
            sum_of_fractions = tf.maximum(sum_of_fractions, 1e-8)
            
            # vf_next: [N, current_num_phases], Σ(vf_next) = 1
            vf_next = vf_next_non_negative / sum_of_fractions
            
            return vf_next
        

    # ========== 辅助方法 ==========
    
    def integrate_pos_vel(self, 
                         pos1: tf.Tensor, 
                         vel1: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        使用半隐式欧拉法进行物理积分。

        Args:
            pos1: 当前位置, shape [N, 3]
            vel1: 当前速度, shape [N, 3]

        Returns:
            (pos2, vel2): 积分后的位置和速度
                pos2: [N, 3]
                vel2: [N, 3]
        
        公式:
            vel2 = vel1 + dt * g
            pos2 = pos1 + dt * (vel1 + vel2) / 2
        """
        dt = self.timestep
        vel2 = vel1 + dt * self.gravity  # [N, 3]
        pos2 = pos1 + dt * (vel1 + vel2) / 2.0  # [N, 3]
        return pos2, vel2

    def compute_new_pos_vel(self, 
                           pos1: tf.Tensor, 
                           vel1: tf.Tensor, 
                           pos2_integrated: tf.Tensor, 
                           vel2_integrated: tf.Tensor, 
                           pos_correction: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        根据网络预测的位置修正量计算最终位置和速度。

        Args:
            pos1: 初始位置, shape [N, 3]
            vel1: 初始速度, shape [N, 3]
            pos2_integrated: 物理积分后的位置, shape [N, 3]
            vel2_integrated: 物理积分后的速度, shape [N, 3]
            pos_correction: 网络预测的修正量, shape [N, 3]

        Returns:
            (pos_final, vel_final): 最终位置和速度
                pos_final: [N, 3]
                vel_final: [N, 3]
        
        公式:
            pos_final = pos2_integrated + pos_correction
            vel_final = (pos_final - pos1) / dt
        """
        dt = self.timestep
        pos_final = pos2_integrated + pos_correction  # [N, 3]
        vel_final = (pos_final - pos1) / dt  # [N, 3]
        return pos_final, vel_final

    def compute_position_correction_from_shared(self, 
                                               shared_features: tf.Tensor, 
                                               pos: tf.Tensor) -> tf.Tensor:
        """
        从共享特征计算位置修正量。

        Args:
            shared_features: 共享特征, shape [N, C]
            pos: 当前位置, shape [N, 3]

        Returns:
            位置修正量, shape [N, 3]
        
        说明:
            修正量乘以 1/128 进行缩放，防止预测过大的修正导致不稳定。
        """
        filter_extent = tf.constant(self.filter_extent)
        
        # pos_conv: [N, 3]
        pos_conv = self.pos_final_conv(shared_features, pos, pos, filter_extent)
        # pos_dense: [N, 3]
        pos_dense = self.pos_final_dense(shared_features)
        
        pos_output = pos_conv + pos_dense
        
        # 缩放修正量（经验值）
        return (1.0 / 128.0) * pos_output

    def init(self, **kwargs) -> None:
        """
        使用虚拟数据初始化模型，以构建网络权重。
        
        该方法会创建一个2相流的虚拟场景来触发网络的第一次前向传播，
        从而完成所有层的权重初始化。
        """
        # ========== 定义虚拟初始化场景 ==========
        init_num_phases = 2  # 使用最常见的2相流进行初始化
        
        if init_num_phases > self.max_num_phases:
            raise ValueError(
                f"Initialization phase count ({init_num_phases}) cannot exceed "
                f"max_num_phases ({self.max_num_phases})."
            )

        # ========== 创建虚拟输入数据 ==========
        # 单个粒子，零值数据
        pos = np.zeros(shape=(1, 3), dtype=np.float32)
        vel = np.zeros(shape=(1, 3), dtype=np.float32)
        
        # 体积分数: 第一相占100%
        phase_fractions = np.zeros(shape=(1, init_num_phases), dtype=np.float32)
        phase_fractions[:, 0] = 1.0

        box = np.zeros(shape=(1, 3), dtype=np.float32)
        box_feats = np.zeros(shape=(1, 3), dtype=np.float32)
        
        # 虚拟密度（水和油的典型值）
        init_densities = np.array([1000.0, 800.0], dtype=np.float32)

        # 虚拟条件参数
        cd = np.float32(0.5)
        cf = np.float32(0.5)
        
        # ========== 执行初始化前向传播 ==========
        print(f"[Model V4] Initializing network weights with {init_num_phases}-phase scenario...")
        _ = self.__call__(
            (pos, vel, phase_fractions, box, box_feats),
            current_num_phases=init_num_phases,
            phase_densities=init_densities,
            cd=cd, 
            cf=cf
        )
        
        # ========== 打印模型信息 ==========
        print(f"\n{'='*80}")
        print(f"[Model V4] Initialization Complete")
        print(f"{'='*80}")
        print(f"  Model name              : {self.name}")
        print(f"  Max phases supported    : {self.max_num_phases}")
        print(f"  Tested with phases      : {init_num_phases}")
        print(f"  Shared feature channels : {self.shared_feature_channels}")
        print(f"  Position output channels: {self.pos_output_channels}")
        print(f"  VF heads created        : {len(self.vf_heads)} configurations")
        print(f"  Total convolutions      : {len(self._all_convs)}")
        print(f"{'='*80}\n")
        
        # 尝试打印模型摘要
        try:
            self.summary()
        except Exception as e:
            print(f"[Warning] Could not print model summary: {e}")