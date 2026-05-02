#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多相流体模拟网络

核心设计：
  1. 位置分支：DeepSet 编码相集合（vf + density）→ 置换不变的全局相嵌入，
       只服务于位置/速度预测，不依赖相数。
  2. VF 分支：独立的 density-free DeepSet 编码当前 VF 集合，
       再结合可选的速度、cd/cf 条件和逐相 vf_i 做权重共享预测，避免密度污染 VF 更新。
  3. cd/cf 条件编码：MLP 将 (cd, cf) 编码为向量，分别注入位置分支和 VF 分支，
       网络通过数据学习混合（cd）与分离（cf）的物理影响。

去掉了 max_num_phases 和动态 VF 分支头——两者均依赖"预先知道相数"，
与 DeepSet 相数无关的设计原则矛盾。
"""

import tensorflow as tf
import open3d.ml.tf as ml3d
import numpy as np
from typing import Tuple, List, Optional
from models.deepset_encoder_v2 import DeepSetPhaseEncoder


class MultiPhaseParticleNetwork(tf.keras.Model):
    """
    数据驱动多相流体模拟网络。

    位置预测：DeepSet(vf+density) + ContinuousConv 主干 → 位置修正头
    VF  预测：density-free VF 分支 + 逐相特征 → 权重共享 MLP → delta_vf → 守恒归一化

    关键设计：VF 逐相预测器的权重在所有相之间共享，
    因此训练时用 2 相，推理时可直接泛化到任意相数，无需 max_num_phases。
    """

    def __init__(self,
                 # 网络结构
                 kernel_size: List[int] = [4, 4, 4],
                 layer_channels: List[int] = [32, 64, 64],
                 # 相特征编码
                 phase_feat_centralization: bool = True,
                 aggregation: str = 'mean',
                 # 条件参数
                 cd_cf_as_input: bool = True,
                 cd_cf_embedding_dim: int = 32,
                 # 物理/仿真参数
                 particle_radius: float = 0.05,
                 radius_scale: float = 1.5,
                 timestep: float = 1 / 50,
                 gravity: Tuple[float, float, float] = (0, -9.81, 0),
                 # 卷积参数
                 coordinate_mapping: str = 'ball_to_cube_volume_preserving',
                 interpolation: str = 'linear',
                 use_window: bool = True,
                 # VF 预测模式
                 # False（默认）= 直接预测：网络每步输出绝对 VF logits → softmax
                 #   优点：无恒等先验，网络必须主动预测，不会出现 VF 不更新问题
                 #   缺点：需要学习绝对 VF 分布，比残差稍难学
                 # True = logit 空间残差：log(vf0) + alpha*delta → softmax
                 #   优点：恒等先验，训练初期稳定
                 #   缺点：delta→0 是有效最优解，可能导致 VF 怠性不更新
                 vf_residual: bool = False,
                 alpha: float = 0.1,
                 vf_use_velocity: bool = False,
                 ) -> None:
        super().__init__(name=type(self).__name__)

        init_vars = locals()
        self.init_params = {k: v for k, v in init_vars.items() if k != 'self'}

        self.layer_channels = layer_channels
        self.phase_feat_centralization = phase_feat_centralization
        self.aggregation = aggregation
        self.cd_cf_as_input = cd_cf_as_input
        self.cd_cf_embedding_dim = cd_cf_embedding_dim
        self.kernel_size = kernel_size
        self.radius_scale = radius_scale
        self.timestep = timestep
        self.gravity = tf.constant(gravity, dtype=tf.float32)
        self.filter_extent = np.float32(radius_scale * 6 * particle_radius)
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.use_window = use_window
        self.vf_residual = vf_residual  # False=直接预测(推荐), True=logit空间残差
        self.alpha = alpha              # Only used when vf_residual=True
        self.vf_use_velocity = vf_use_velocity

        self._all_convs = []

        def window_poly6(r_sqr):
            return tf.clip_by_value((1 - r_sqr) ** 3, 0, 1)

        def Conv(name, filters, activation=None, **kwargs):
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
                **kwargs,
            )
            self._all_convs.append((name, conv))
            return conv

        # ── DeepSet 相编码器 ────────────────────────────────────────────────
        # 输入: [N, num_phases, 2]  输出: [N, 64]
        # 对相集合置换不变，相数可变
        self.phase_encoder = DeepSetPhaseEncoder(
            phi_dims=[64, 128],
            rho_dims=[128, 64],
            aggregation=self.aggregation,
        )

        # ── VF 分支专用 DeepSet 编码器（仅看 VF，不看密度）──────────────────────
        # 输入: [N, num_phases, 1]  输出: [N, 64]
        # 仅为 VF 更新构建 density-free 的置换不变上下文
        self.vf_phase_encoder = DeepSetPhaseEncoder(
            phi_dims=[64, 128],
            rho_dims=[128, 64],
            aggregation=self.aggregation,
        )

        # ── cd/cf 条件编码器 ────────────────────────────────────────────────
        if self.cd_cf_as_input:
            self.cd_cf_encoder = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', name='cd_cf_enc1'),
                tf.keras.layers.Dense(cd_cf_embedding_dim, activation='relu', name='cd_cf_enc2'),
            ], name='cd_cf_encoder')
            _ = self.cd_cf_encoder(tf.zeros((1, 2), dtype=tf.float32))

        # ── 主干网络 ────────────────────────────────────────────────────────
        self.conv0_fluid = Conv('conv0_fluid', filters=layer_channels[0])
        self.conv0_obstacle = Conv('conv0_obstacle', filters=layer_channels[0])
        self.dense0_fluid = tf.keras.layers.Dense(units=layer_channels[0], name='dense0_fluid')

        self.convs: List[ml3d.layers.ContinuousConv] = []
        self.denses: List[tf.keras.layers.Dense] = []
        for i, ch in enumerate(layer_channels[1:], 1):
            self.denses.append(tf.keras.layers.Dense(units=ch, name=f'dense{i}'))
            self.convs.append(Conv(f'conv{i}', filters=ch))

        # ── 位置修正预测头 ──────────────────────────────────────────────────
        self.pos_conv = Conv('pos_conv', filters=3)
        self.pos_dense = tf.keras.layers.Dense(units=3, name='pos_dense')

        # ── VF 空间卷积（捕捉相分数在邻域间的传播）─────────────────────────
        # 输入: density-free VF branch features [N, C]  输出: [N, C]
        # 卷积聚合邻域 VF 状态，不再复用位置分支的 density-conditioned latent
        self.vf_context_conv = Conv('vf_context_conv', filters=layer_channels[-1])

        # ── VF 逐相预测器（权重共享 MLP）───────────────────────────────────
        # 输入: [N, num_phases, layer_channels[-1] + 1 (+ cd_cf_dim)]
        #   layer_channels[-1]: VF 空间卷积输出（聚合了邻域相分数信息）
        #   1: 归一化后的 vf_i（仅 VF，不含密度——密度只留在位置分支）
        # 输出: [N, num_phases, 1] → squeeze → [N, num_phases]
        #
        # 同一 MLP 对所有相（所有粒子）权重共享，等同于逐相独立前向传播。
        # 因此训练时 num_phases=2，推理时 num_phases=3/4/N 均可直接使用，
        # 网络学到的是"如何根据全局状态预测单个相的变化"，与相总数无关。
        self.vf_per_phase_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', name='vf_h1'),
            tf.keras.layers.Dense(32, activation='relu', name='vf_h2'),
            tf.keras.layers.Dense(1, name='vf_out'),
        ], name='vf_per_phase_mlp')

    # ─────────────────────────────────────────────────────────────────────────
    #  call
    # ─────────────────────────────────────────────────────────────────────────

    def call(self,
             inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
             current_num_phases: Optional[tf.Tensor] = None,
             phase_densities: Optional[tf.Tensor] = None,
             training: bool = False,
             **kwargs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        前向传播。

        Args:
            inputs: (pos1, vel1, current_phase_fractions, box_pos, box_feats)
                pos1                   : [N, 3]
                vel1                   : [N, 3]
                current_phase_fractions: [N, num_phases]  num_phases 可变
                box_pos                : [M, 3]
                box_feats              : [M, F]
            current_num_phases: 兼容旧接口保留，实际相数从输入张量形状读取
            phase_densities   : [num_phases] 各相密度；None 时全取 1000 kg/m³
            training          : 训练模式标志
            **kwargs          : cd (float), cf (float)

        Returns:
            (pos_final [N,3], vel_final [N,3], next_phase_fractions [N, num_phases])
        """
        pos1, vel1, current_phase_fractions, box_pos, box_feats = inputs

        # 实际相数直接从张量形状读取，不依赖 current_num_phases
        num_phases = tf.shape(current_phase_fractions)[1]

        if phase_densities is None:
            phase_densities = tf.ones([num_phases], dtype=tf.float32) * 1000.0
        phase_densities = tf.convert_to_tensor(phase_densities, dtype=tf.float32)

        # 1. 物理积分
        pos2, vel2 = self.integrate_pos_vel(pos1, vel1)

        # 2. 相特征归一化（位置分支 DeepSet 与 VF 分支共享原始逐相输入）
        per_phase_features, phase_embedding = self._encode_phases(
            current_phase_fractions, phase_densities, num_phases
        )
        vf_only_features, vf_phase_embedding = self._encode_vf_branch(
            per_phase_features, num_phases
        )

        # 3. cd/cf 编码一次，主干网络和 VF 头共用同一嵌入向量
        #    这样 VF 头对 cd/cf 有直接的短路梯度路径，不需要穿越整个主干
        cd_cf_emb = None
        if self.cd_cf_as_input:
            # 显式校验，不传直接抛异常
            if 'cd' not in kwargs or 'cf' not in kwargs:
                raise ValueError("cd, cf must be provided in kwargs when cd_cf_as_input is True")
            # 严格必传：不传直接 KeyError 报错，程序停止
            cd = tf.cast(kwargs['cd'], dtype=tf.float32)
            cf = tf.cast(kwargs['cf'], dtype=tf.float32)
            cond = tf.reshape(tf.stack([cd, cf]), (1, 2))
            cd_cf_raw = self.cd_cf_encoder(cond)                        # [1, D]
            cd_cf_emb = tf.tile(cd_cf_raw, [tf.shape(pos2)[0], 1])     # [N, D]

        # 4. 构建粒子特征（主干网络用）
        fluid_feats = self._build_fluid_feats(pos2, vel2, phase_embedding, cd_cf_emb)

        # 5. 主干网络
        shared_features = self._backbone(fluid_feats, pos2, box_pos, box_feats)

        # 6. 位置修正
        filter_extent = tf.constant(self.filter_extent)
        pos_correction = (1.0 / 128.0) * (
            self.pos_conv(shared_features, pos2, pos2, filter_extent,
                          user_neighbors_index=self._fluid_nns_index,
                          user_neighbors_row_splits=self._fluid_nns_row_splits,
                          user_neighbors_importance=self._fluid_nns_importance)
            + self.pos_dense(shared_features)
        )
        pos_final, vel_final = self.compute_new_pos_vel(pos1, vel1, pos2, vel2, pos_correction)

        # 7. VF 分支独立上下文：只看可选 velocity + VF-only DeepSet + cd/cf
        vf_branch_feats = self._build_vf_feats(vel_final, vf_phase_embedding, cd_cf_emb)

        # 8. VF 逐相预测（与位置分支解耦，避免密度经 shared_features 污染 VF）
        next_vf = self._predict_next_vf(
            vf_branch_feats, vf_only_features, current_phase_fractions, num_phases, pos_final,
            cd_cf_emb=cd_cf_emb
        )

        return pos_final, vel_final, next_vf

    # ─────────────────────────────────────────────────────────────────────────
    #  相特征编码
    # ─────────────────────────────────────────────────────────────────────────

    def _encode_phases(self,
                       phase_fractions: tf.Tensor,
                       phase_densities: tf.Tensor,
                       num_phases: tf.Tensor):
        """
        归一化相特征，返回：
          per_phase_features: [N, num_phases, 2]  逐相特征（供逐相预测器使用）
          phase_embedding   : [N, 64]             DeepSet 聚合嵌入（供主干网络使用）
        """
        densities_per_particle = tf.broadcast_to(
            phase_densities[:num_phases], tf.shape(phase_fractions)
        )
        log_densities = tf.math.log(densities_per_particle + 1e-8)

        if self.phase_feat_centralization:
            vf_scaled = (phase_fractions - 0.5) * 2.0
            log_density_scaled = (log_densities - 7.7) / 1.5
        else:
            vf_scaled = phase_fractions
            log_density_scaled = (log_densities - 6.2146) / 2.9957

        per_phase_features = tf.stack([vf_scaled, log_density_scaled], axis=-1)  # [N, P, 2]

        N = tf.shape(phase_fractions)[0]
        mask = tf.ones([N, num_phases], dtype=tf.bool)
        phase_embedding = self.phase_encoder((per_phase_features, mask))  # [N, 64]

        return per_phase_features, phase_embedding

    def _encode_vf_branch(self,
                          per_phase_features: tf.Tensor,
                          num_phases: tf.Tensor):
        """
        为 VF 分支构建仅含 VF 的置换不变编码。

        返回：
          vf_only_features : [N, num_phases, 1]  逐相 vf_scaled
          vf_phase_embedding: [N, 64]            DeepSet(VF-only) 聚合嵌入
        """
        vf_only_features = per_phase_features[..., 0:1]

        N = tf.shape(per_phase_features)[0]
        mask = tf.ones([N, num_phases], dtype=tf.bool)
        vf_phase_embedding = self.vf_phase_encoder((vf_only_features, mask))  # [N, 64]

        return vf_only_features, vf_phase_embedding

    # ─────────────────────────────────────────────────────────────────────────
    #  粒子特征构建
    # ─────────────────────────────────────────────────────────────────────────

    def _build_fluid_feats(self,
                           pos: tf.Tensor,
                           vel: tf.Tensor,
                           phase_embedding: tf.Tensor,
                           cd_cf_emb: Optional[tf.Tensor] = None) -> tf.Tensor:
        """[N, 1 + 3 + 64 + cd_cf_dim]"""
        feats = [
            tf.ones_like(pos[:, 0:1]),  # [N, 1]
            vel,                         # [N, 3]
            phase_embedding,             # [N, 64]
        ]
        if cd_cf_emb is not None:
            feats.append(cd_cf_emb)      # [N, D]
        return tf.concat(feats, axis=-1)

    def _build_vf_feats(self,
                        vel: tf.Tensor,
                        vf_phase_embedding: tf.Tensor,
                        cd_cf_emb: Optional[tf.Tensor] = None) -> tf.Tensor:
        """[N, 1 + (3 if vf_use_velocity else 0) + 64 + cd_cf_dim]，仅供 VF 分支使用，不含密度。"""
        feats = [
            tf.ones_like(vel[:, 0:1]),  # [N, 1]
            vf_phase_embedding,         # [N, 64]
        ]
        if self.vf_use_velocity:
            feats.insert(1, vel)        # [N, 3]
        if cd_cf_emb is not None:
            feats.append(cd_cf_emb)     # [N, D]
        return tf.concat(feats, axis=-1)

    # ─────────────────────────────────────────────────────────────────────────
    #  主干网络
    # ─────────────────────────────────────────────────────────────────────────

    def _backbone(self,
                  fluid_feats: tf.Tensor,
                  pos: tf.Tensor,
                  box_pos: tf.Tensor,
                  box_feats: tf.Tensor) -> tf.Tensor:
        filter_extent = tf.constant(self.filter_extent)

        x = tf.concat([
            self.conv0_obstacle(box_feats, box_pos, pos, filter_extent),
            self.conv0_fluid(fluid_feats, pos, pos, filter_extent),
            self.dense0_fluid(fluid_feats),
        ], axis=-1)

        # 缓存 pos→pos 邻居搜索结果，后续所有 fluid-fluid 卷积（同一 filter_extent）复用，
        # 避免重复 radius search（backbone 循环卷积 + pos_conv 共享此缓存）
        self._fluid_nns_index = self.conv0_fluid.nns.neighbors_index
        self._fluid_nns_row_splits = self.conv0_fluid.nns.neighbors_row_splits
        self._fluid_nns_importance = self.conv0_fluid._conv_values['neighbors_importance']

        self.num_fluid_neighbors = ml3d.ops.reduce_subarrays_sum(
            tf.ones_like(self.conv0_fluid.nns.neighbors_index, dtype=tf.float32),
            self.conv0_fluid.nns.neighbors_row_splits,
        )

        ans = [x]
        for conv, dense in zip(self.convs, self.denses):
            inp = tf.keras.activations.relu(ans[-1])
            out = conv(inp, pos, pos, filter_extent,
                       user_neighbors_index=self._fluid_nns_index,
                       user_neighbors_row_splits=self._fluid_nns_row_splits,
                       user_neighbors_importance=self._fluid_nns_importance) + dense(inp)
            if out.shape[-1] == ans[-1].shape[-1]:
                out = out + ans[-1]
            ans.append(out)

        return tf.keras.activations.relu(ans[-1])

    # ─────────────────────────────────────────────────────────────────────────
    #  VF 逐相预测
    # ─────────────────────────────────────────────────────────────────────────

    def _predict_next_vf(self,
                         vf_branch_feats: tf.Tensor,
                         vf_only_features: tf.Tensor,
                         current_vf: tf.Tensor,
                         num_phases: tf.Tensor,
                         pos: tf.Tensor,
                         cd_cf_emb: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        逐相预测 VF 更新，权重在所有相之间共享。

        两种模式（由 self.vf_residual 控制）：
          直接预测（vf_residual=False，默认）：
            vf_next = softmax(MLP_output)
            网络每步必须显式预测完整 VF 分布，无法通过输出 0 敷衍，
            能有效避免"VF 不更新"问题。

          logit 空间残差（vf_residual=True）：
            vf_next = softmax(log(vf0 + ε) + alpha * MLP_output)
            恒等先验：MLP 输出为 0 时 vf_next = vf0，训练初期稳定，
            但存在网络学到 delta→0 懒惰最优解的风险。

        两种模式均满足 vf ≥ 0 且 Σvf = 1（softmax 保证），无 tanh 饱和。
        VF 分支使用独立的 density-free 上下文，密度只通过位置更新后的邻域变化间接影响 VF。
        cd/cf_emb 直接拼接，为混溶/分离条件提供短路梯度路径。
        """
        filter_extent = tf.constant(self.filter_extent)

        # 空间卷积：让每个粒子在新位置处感知邻域 VF 状态
        vf_spatial = self.vf_context_conv(vf_branch_feats, pos, pos, filter_extent)  # [N, C]
        vf_spatial_expanded = tf.tile(
            tf.expand_dims(vf_spatial, axis=1), [1, num_phases, 1]
        )  # [N, num_phases, C]

        # 拼接：局部空间特征 + 逐相 VF（仅 vf_scaled）+ cd/cf 条件
        # 密度不进入 VF 分支；分层只通过 pos_final 改变的邻域结构间接体现
        parts = [vf_spatial_expanded, vf_only_features]  # [N, P, C+1]
        if cd_cf_emb is not None:
            cd_cf_expanded = tf.tile(
                tf.expand_dims(cd_cf_emb, axis=1), [1, num_phases, 1]
            )  # [N, P, D]
            parts.append(cd_cf_expanded)
        per_phase_input = tf.concat(parts, axis=-1)  # [N, P, C+1(+D)]

        delta_logits = tf.squeeze(self.vf_per_phase_mlp(per_phase_input), axis=-1)  # [N, P]

        if self.vf_residual:
            # logit 空间残差：log(vf0) + alpha * delta_logits → softmax
            # 恒等先验：delta_logits=0 时 vf_next = vf0
            # 风险：网络可能学到 delta→0 的怠性最优解（VF 几乎不更新）
            vf_logits = tf.math.log(current_vf + 1e-8) + self.alpha * delta_logits
        else:
            # 直接预测：网络输出绝对 VF logits，无恒等先验
            # 网络必须每步显式预测 VF 分布，不能通过 delta→0 敷衍
            vf_logits = delta_logits
        return tf.nn.softmax(vf_logits, axis=-1)

    # ─────────────────────────────────────────────────────────────────────────
    #  辅助方法
    # ─────────────────────────────────────────────────────────────────────────

    def integrate_pos_vel(self,
                          pos1: tf.Tensor,
                          vel1: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        dt = self.timestep
        vel2 = vel1 + dt * self.gravity
        pos2 = pos1 + dt * (vel1 + vel2) / 2.0
        return pos2, vel2

    def compute_new_pos_vel(self,
                            pos1: tf.Tensor,
                            vel1: tf.Tensor,
                            pos2_integrated: tf.Tensor,
                            vel2_integrated: tf.Tensor,
                            pos_correction: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        dt = self.timestep
        pos_final = pos2_integrated + pos_correction
        vel_final = (pos_final - pos1) / dt
        return pos_final, vel_final

    # ─────────────────────────────────────────────────────────────────────────
    #  初始化
    # ─────────────────────────────────────────────────────────────────────────

    def init(self, **kwargs) -> None:
        """用 2 相虚拟数据触发前向传播，完成所有权重初始化。"""
        pos = np.zeros((1, 3), dtype=np.float32)
        vel = np.zeros((1, 3), dtype=np.float32)
        phase_fractions = np.array([[1.0, 0.0]], dtype=np.float32)
        box = np.zeros((1, 3), dtype=np.float32)
        box_feats = np.zeros((1, 3), dtype=np.float32)
        densities = np.array([1000.0, 800.0], dtype=np.float32)

        _ = self.__call__(
            (pos, vel, phase_fractions, box, box_feats),
            phase_densities=tf.constant(densities, dtype=tf.float32),
            cd=np.float32(0.5),
            cf=np.float32(0.5),
        )
