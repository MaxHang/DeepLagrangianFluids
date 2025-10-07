"""
版本特点 (V2 compared to V1):
1.  **支持可变相数 (Variable Number of Phases)**: 模型不再局限于一个固定的相数。
    将 DeepSetPhaseEncoder 集成到主网络中

3.  **残差预测 (Residual Prediction) **: 预测体积分数的变化量 (delta_logits)
"""

import tensorflow as tf
import open3d.ml.tf as ml3d
import numpy as np
from debug_utils import debug_print
from models.deepset_encoder import DeepSetPhaseEncoder

import time

class MultiPhaseParticleNetwork(tf.keras.Model):

    def __init__(self,
                 # --- 核心改动：使用 max_num_phases ---
                 max_num_phases=5, # 模型能处理的最大相数
                 # ------------------------------------
                 kernel_size=[4, 4, 4],
                 radius_scale=1.5,
                 coordinate_mapping='ball_to_cube_volume_preserving',
                 interpolation='linear',
                 aggregation='mean',  # 新增：deepset聚合方式
                 phase_feat_centralization=False,  # 新增：是否对特征进行中心化
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
        self.aggregation = aggregation
        self.phase_feat_centralization = phase_feat_centralization

        debug_print(f"Model initialized to handle up to {self.max_num_phases} phases.")
        debug_print(f"using {self.aggregation} aggregation method for DeepSetPhaseEncoder.")
        debug_print(f"Using {self.phase_feat_centralization} feature centralization.")

        if self.cd_cf_as_input and self.cd_cf_embedding_dim > 0:
            self.cd_embedding_layer = tf.keras.layers.Dense(self.cd_cf_embedding_dim, activation='tanh', name='cd_embedding')
            self.cf_embedding_layer = tf.keras.layers.Dense(self.cd_cf_embedding_dim, activation='tanh', name='cf_embedding')

        # 创建 Deep Sets 编码器实例
        self.phase_encoder = DeepSetPhaseEncoder(
            phi_dims=[64, 128],  # 示例维度
            rho_dims=[128, 64],   # 示例维度
            aggregation=self.aggregation,  # 使用指定的聚合方式
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
        debug_print("pos1 1000: \n", pos1[::1000])
        debug_print("vel1 1000: \n", vel1[::1000])
        debug_print("current_phase_fractions 1000: \n", current_phase_fractions[::1000])
        
        phase_densities = tf.convert_to_tensor(phase_densities, dtype=tf.float32)

        # 1. 积分位置和速度（基础版本）
        pos2_integrated, vel2_integrated = self.integrate_pos_vel(pos1, vel1)
        debug_print("pos2_integrated 1000: \n", pos2_integrated[::1000])

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
        
        debug_print("shared_features 1000: \n", shared_features[::1000])

        # 3. 基于共享特征预测位置修正
        pos_correction = self.compute_position_correction_from_shared(shared_features, pos2_integrated)

        debug_print("pos_correction 1000: \n", pos_correction[::1000])

        # 4. 应用位置修正并计算最终速度
        pos_final, vel_final = self.compute_new_pos_vel(pos1, vel1, pos2_integrated, vel2_integrated, pos_correction)

        # 5. 基于共享特征预测相分数
        next_phase_fractions_final = current_phase_fractions
        # a. 预测体积分数的变化量
        predicted_delta_vf = self.compute_delta_vf_from_shared(shared_features, pos_final)

        debug_print("predicted_delta_vf 1000: \n", predicted_delta_vf[::1000])
        
        # b. 准备残差连接所需的、经过填充的当前体积分数
        padding_size = self.max_num_phases - current_num_phases
        paddings = [[0, 0], [0, padding_size]] # 假设VF形状是 [N, current_num_phases]
        vf_current_padded = tf.pad(current_phase_fractions, paddings, "CONSTANT", constant_values=0)
        
        # c. 调用新的残差计算函数
        vf_next_padded = self.compute_next_phase_fractions(
            vf_current_padded, 
            predicted_delta_vf,
            current_num_phases
        )

        debug_print("vf_next_padded 1000: \n", vf_next_padded[::1000])
        
        # d. 切片，得到最终的、符合当前相数的输出 (这一步仍然是必要的)
        next_phase_fractions_final = vf_next_padded[:, :current_num_phases]
        
        debug_print("pos_final 1000: \n ", pos_final[::1000])
        debug_print("vel_final 1000: \n ", vel_final[::1000])
        debug_print("next_phase_fractions_final 1000: \n", next_phase_fractions_final[::1000])

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

        # b. 进行特定于数据的预处理
        # b.1. 对数变换密度
        log_densities_per_particle = tf.math.log(densities_per_particle + 1e-8)


        if self.phase_feat_centralization:
            # [-1, 1]
            vf_scaled = (phase_fractions - 0.5) * 2.0
            # 2. 定义基于您数据范围 [500, 10000] 的中心和缩放常数
            LOG_DENSITY_CENTER = 7.7  # log(sqrt(500 * 10000))
            LOG_DENSITY_SCALE = 1.5   # (log(10000) - log(500)) / 2

            # 3. 中心化与缩放
            log_density_scaled = (log_densities_per_particle - LOG_DENSITY_CENTER) / LOG_DENSITY_SCALE
        else:
            # b.2. 体积分数 (vf) 已经是 [0, 1] 范围，我们可以保持原样，或者进行一个小的变换
            #    例如，将其中心移到0附近：vf_scaled = (phase_fractions - 0.5) * 2.0  # -> [-1, 1]
            vf_scaled = phase_fractions # 保持 [0, 1] 通常也可以
            # b.3. 对数密度 (log_density)
            # 定义基于您数据范围 [500, 10000] 的对数最小值和范围
            LOG_DENSITY_MIN = 6.2146  # log(500)
            LOG_DENSITY_RANGE = 2.9957 # log(10000) - log(500)

            # Min-Max 缩放
            log_density_scaled = (log_densities_per_particle - LOG_DENSITY_MIN) / LOG_DENSITY_RANGE
            # 这个操作会确保输出严格在 [0, 1] 范围内 (可能会因 epsilon 略有偏差)
            log_density_scaled = tf.clip_by_value(log_density_scaled, 0.0, 1.0) # 可选，增加稳定性
        
        # b.2. 拼接成完整的相特征向量 [vf, log_density]
        per_phase_features = tf.stack([vf_scaled, log_density_scaled], axis=-1)

        debug_print("per_phase_features 1000: \n", per_phase_features[::1000])

        # c. 进行填充以匹配 max_num_phases
        padding_size = self.max_num_phases - current_num_phases
        # paddings 格式: [[axis0_pad], [axis1_pad], [axis2_pad], [axis3_pad]]
        paddings = [[0, 0], [0, padding_size], [0, 0]]
        padded_features = tf.pad(per_phase_features, paddings, "CONSTANT")
        
        # d. 创建掩码
        mask_range = tf.range(self.max_num_phases)
        mask = mask_range < current_num_phases # 形状: [max_num_phases]
        # 广播到每个粒子上
        mask_per_particle = tf.broadcast_to(mask, [tf.shape(pos)[0], self.max_num_phases])

        debug_print("Shape of padded_features: ", tf.shape(padded_features))
        debug_print("Shape of mask_per_particle: ", tf.shape(mask_per_particle))

        # --- 2. 使用 Deep Sets 编码器 ---
        debug_print("particle_features 1000: \n", padded_features[::1000])
        deep_sets_start = time.time()
        particle_phase_embedding = self.phase_encoder([padded_features, mask_per_particle])
        deep_sets_time = time.time() - deep_sets_start
        debug_print("particle_phase_embedding 1000: \n", particle_phase_embedding[::1000])

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

        if training and tf.random.uniform([]) < 0.01:  # 1%的概率打印时间
            debug_print("Deep Sets time: ", deep_sets_time)
            debug_print("Total: ", total_time)

        return tf.keras.activations.relu(shared_ans_convs[-1])

    def compute_next_phase_fractions(self, current_phase_fractions_padded, network_delta_vf_padded, current_num_phases):
        """
        在概率空间进行残差预测，并通过重新归一化来保证有效性。
        这是一个数值上比 log 空间更稳健的方案。

        Args:
            current_phase_fractions_padded (tf.Tensor): [N, max_num_phases], 填充后的当前VF。
            network_delta_vf_padded (tf.Tensor): [N, max_num_phases], 网络预测的VF变化量 (范围在[-1, 1])。
            current_num_phases (tf.Tensor): 标量整数, 当前有效的相数。
        """
        # 1. 直接在概率空间进行残差连接
        #    用一个小的缩放因子来控制每个时间步的最大变化幅度，增加稳定性。
        #    这个 0.1 是一个超参数，可以根据需要调整。
        vf_next_unnormalized = current_phase_fractions_padded + 0.1 * network_delta_vf_padded

        # 2. 确保结果非负 (概率不能是负数)
        vf_next_non_negative = tf.keras.activations.relu(vf_next_unnormalized)
        
        # 3. 重新归一化 (关键步骤)
        # a. 创建掩码，只在有效区域操作
        mask_range = tf.range(self.max_num_phases, dtype=tf.int32)
        phase_mask = mask_range < current_num_phases
        phase_mask_float = tf.cast(tf.broadcast_to(phase_mask, tf.shape(vf_next_non_negative)), dtype=tf.float32)

        # b. 将无效区域（填充区）严格置零
        vf_next_masked = vf_next_non_negative * phase_mask_float

        # c. 计算每个粒子有效相的总和
        sum_of_fractions = tf.reduce_sum(vf_next_masked, axis=1, keepdims=True)
        
        # d. 避免除以零 (如果一个粒子所有相的vf都变成0，让它保持为0)
        sum_of_fractions = tf.maximum(sum_of_fractions, 1e-8)
        
        # e. 执行归一化，得到最终的、和为1的概率分布
        vf_next_normalized_padded = vf_next_masked / sum_of_fractions
        
        return vf_next_normalized_padded


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
        pos_conv = self.pos_final_conv(shared_features, pos, pos, filter_extent_tensor)
        pos_dense = self.pos_final_dense(shared_features)
        pos_output = pos_conv + pos_dense
        pos_correction = (1.0 / 128.0) * pos_output
        # debug_print("Shape of position correction: ", tf.shape(pos_correction))
        return pos_correction

    def compute_delta_vf_from_shared(self, shared_features, pos):
        if self.max_num_phases <= 1:
            return None
        filter_extent_tensor = tf.constant(self.filter_extent, dtype=tf.float32)
        vf_conv = self.vf_final_conv(shared_features, pos, pos, filter_extent_tensor)
        vf_dense = self.vf_final_dense(shared_features)
        vf_logits = vf_conv + vf_dense
        # debug_print("Shape of vf_logits from shared network: ", tf.shape(vf_logits))
        return tf.keras.activations.tanh(vf_logits)  # 使用 tanh 激活函数
    
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
            print(f"VF output channels (max): {self.vf_output_channels}")
        
        try:
            # 打印模型摘要
            self.summary()
        except Exception as e:
            print(f"Could not print model summary: {e}")

