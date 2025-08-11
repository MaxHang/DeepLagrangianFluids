import tensorflow as tf
import open3d.ml.tf as ml3d
import numpy as np
from debug_utils import debug_print


class MultiPhaseParticleNetwork(tf.keras.Model):

    def __init__(self,
                 kernel_size=[4, 4, 4],
                 radius_scale=1.5,
                 coordinate_mapping='ball_to_cube_volume_preserving',
                 interpolation='linear',
                 use_window=True,
                 particle_radius=0.05,
                 timestep=1 / 50,
                 gravity=(0, -9.81, 0),
                 num_phases=2,
                 cd_cf_as_input=True,
                 cd_cf_embedding_dim=16):
        super().__init__(name=type(self).__name__)

        self.num_phases = num_phases

        self.shared_feature_channels = [32, 64, 64]  # 共享的特征提取层
        # 特定任务的输出层
        self.pos_output_channels = 3  # 位置修正
        self.vf_output_channels = self.num_phases if num_phases > 1 else 0  # 相分数logits

        self.kernel_size = kernel_size
        self.radius_scale = radius_scale
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.use_window = use_window
        self.particle_radius = particle_radius
        self.filter_extent = np.float32(self.radius_scale * 6 * self.particle_radius)
        self.timestep = timestep
        self.gravity = tf.constant(gravity, dtype=tf.float32) # Make gravity a constant tensor

        self.cd_cf_as_input = cd_cf_as_input
        self.cd_cf_embedding_dim = cd_cf_embedding_dim

        debug_print(f"Particle Radius: {self.particle_radius}")
        debug_print(f"Filter Extent: {self.filter_extent}")
        debug_print(f"Number of Phases: {self.num_phases}")
        debug_print(f"Shared feature channels: {self.shared_feature_channels}")
        debug_print(f"Position output channels: {self.pos_output_channels}")
        if self.num_phases > 1:
            debug_print(f"VF output channels: {self.vf_output_channels}")


        if self.cd_cf_as_input and self.cd_cf_embedding_dim > 0:
            self.cd_embedding_layer = tf.keras.layers.Dense(self.cd_cf_embedding_dim,
                                                            activation='tanh', # Or 'relu' or None
                                                            name='cd_embedding')
            self.cf_embedding_layer = tf.keras.layers.Dense(self.cd_cf_embedding_dim,
                                                            activation='tanh', # Or 'relu' or None
                                                            name='cf_embedding')

        self._all_convs = []

        def window_poly6(r_sqr):
            # Ensure r_sqr is not negative, which can happen due to precision issues
            r_sqr_clipped = tf.maximum(r_sqr, 0.0)
            return tf.clip_by_value((1 - r_sqr_clipped)**3, 0, 1)


        def Conv(name, activation=None, **kwargs):
            conv_fn = ml3d.layers.ContinuousConv
            window_fn = None
            if self.use_window:
                window_fn = window_poly6

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

        # 共享特征提取网络层
        self.shared_conv0_fluid = Conv(name="shared_conv0_fluid",
                                      filters=self.shared_feature_channels[0],
                                      activation=None)
        self.shared_conv0_obstacle = Conv(name="shared_conv0_obstacle",
                                         filters=self.shared_feature_channels[0],
                                         activation=None)
        self.shared_dense0_fluid = tf.keras.layers.Dense(name="shared_dense0_fluid",
                                                        units=self.shared_feature_channels[0],
                                                        activation=None)

        # 共享的中间层
        self.shared_convs = []
        self.shared_denses = []
        for i in range(1, len(self.shared_feature_channels)):
            ch = self.shared_feature_channels[i]
            dense_name = f"shared_dense{i}"
            conv_name = f"shared_conv{i}"
            
            dense = tf.keras.layers.Dense(units=ch, name=dense_name, activation=None)
            conv = Conv(name=conv_name, filters=ch, activation=None)
            self.shared_denses.append(dense)
            self.shared_convs.append(conv)

        # 位置预测专用输出层
        self.pos_final_conv = Conv(name="pos_final_conv", 
                                  filters=self.pos_output_channels, 
                                  activation=None)
        self.pos_final_dense = tf.keras.layers.Dense(units=self.pos_output_channels,
                                                    name="pos_final_dense",
                                                    activation=None)

        # 相分数预测专用输出层（仅在多相时创建）
        if self.num_phases > 1:
            self.vf_final_conv = Conv(name="vf_final_conv",
                                     filters=self.vf_output_channels,
                                     activation=None)
            self.vf_final_dense = tf.keras.layers.Dense(units=self.vf_output_channels,
                                                       name="vf_final_dense",
                                                       activation=None)

    def call(self, inputs, training=False, fixed_radius_search_hash_table=None, cd=0.5, cf=0.5):
        pos1, vel1, current_phase_fractions, box_pos, box_feats = inputs

        # 1. 积分位置和速度
        pos2_integrated, vel2_integrated = self.integrate_pos_vel(pos1, vel1)

        # 2. 计算共享特征（这是最耗时的部分）
        shared_features = self.compute_shared_features(
            pos2_integrated,
            vel2_integrated,
            current_phase_fractions,
            box_pos,
            box_feats,
            fixed_radius_search_hash_table,
            cd_scalar=cd,
            cf_scalar=cf)

        # 3. 基于共享特征预测位置修正
        pos_correction = self.compute_position_correction_from_shared(
            shared_features, pos2_integrated)

        # 4. 应用位置修正并计算最终速度
        pos_final, vel_final = self.compute_new_pos_vel(
            pos1, vel1, pos2_integrated, vel2_integrated, pos_correction)

        # 5. 基于共享特征和修正后的位置预测相分数logits
        next_vf_logits = None
        if self.num_phases > 1:
            # 注意：这里使用修正后的位置进行相分数预测
            next_vf_logits = self.compute_vf_logits_from_shared(
                shared_features, pos_final)

        # 6. 计算下一时步的相分数
        next_phase_fractions_final = current_phase_fractions
        if self.num_phases > 1 and next_vf_logits is not None:
            next_phase_fractions_final = self.compute_next_phase_fractions(
                current_phase_fractions,
                next_vf_logits)
        
        return pos_final, vel_final, next_phase_fractions_final
    
    def compute_shared_features(self,
                               pos,
                               vel,
                               phase_fractions,
                               box_pos,
                               box_feats,
                               fixed_radius_search_hash_table=None,
                               cd_scalar=0.5,
                               cf_scalar=0.5):
        """计算位置和相分数预测的共享特征"""
        filter_extent_tensor = tf.constant(self.filter_extent, dtype=tf.float32)

        # --- 特征工程 ---
        fluid_feats_list = [
            tf.ones_like(pos[:, 0:1]),  # 存在特征
            vel  # 当前速度
        ]

        if self.num_phases > 1 and phase_fractions is not None:
            fluid_feats_list.append(phase_fractions)  # 原始相分数

        # 添加cd/cf特征
        if self.cd_cf_as_input:
            cd_tensor_val = tf.cast(cd_scalar, dtype=tf.float32)
            cf_tensor_val = tf.cast(cf_scalar, dtype=tf.float32)
            batch_size = tf.shape(pos)[0]

            if self.cd_cf_embedding_dim > 0:
                cd_per_particle = tf.ones((batch_size, 1), dtype=tf.float32) * cd_tensor_val
                cf_per_particle = tf.ones((batch_size, 1), dtype=tf.float32) * cf_tensor_val
                
                cd_embed = self.cd_embedding_layer(cd_per_particle)
                cf_embed = self.cf_embedding_layer(cf_per_particle)
                fluid_feats_list.extend([cd_embed, cf_embed])
            else:
                cd_direct = tf.ones_like(pos[:, 0:1]) * cd_tensor_val
                cf_direct = tf.ones_like(pos[:, 0:1]) * cf_tensor_val
                fluid_feats_list.extend([cd_direct, cf_direct])

        fluid_feats = tf.concat(fluid_feats_list, axis=-1)
        debug_print("Shape of fluid_feats for shared features: ", tf.shape(fluid_feats))

        # --- 共享网络前向传播 ---
        # 第一层
        shared_conv0_fluid = self.shared_conv0_fluid(fluid_feats, pos, pos, filter_extent_tensor)
        shared_dense0_fluid = self.shared_dense0_fluid(fluid_feats)
        shared_conv0_obstacle = self.shared_conv0_obstacle(box_feats, box_pos, pos, filter_extent_tensor)
        
        processed_feats = tf.concat([shared_conv0_obstacle, shared_conv0_fluid, shared_dense0_fluid], axis=-1)

        # 中间层
        shared_ans_convs = [processed_feats]
        for conv_layer, dense_layer in zip(self.shared_convs, self.shared_denses):
            current_features = tf.keras.activations.relu(shared_ans_convs[-1])
            
            ans_conv = conv_layer(current_features, pos, pos, filter_extent_tensor)
            ans_dense = dense_layer(current_features)
            
            # 残差连接
            if ans_dense.shape[-1] == shared_ans_convs[-1].shape[-1]:
                ans = ans_conv + ans_dense + shared_ans_convs[-1]
            else:
                ans = ans_conv + ans_dense
            shared_ans_convs.append(ans)

        # 返回共享特征
        shared_features = shared_ans_convs[-1]
        
        # 存储邻居数量信息（用于损失计算）
        self.num_fluid_neighbors = ml3d.ops.reduce_subarrays_sum(
            tf.ones_like(self.shared_conv0_fluid.nns.neighbors_index, dtype=tf.float32),
            self.shared_conv0_fluid.nns.neighbors_row_splits)

        return shared_features
    
    def compute_next_phase_fractions(self, current_phase_fractions, network_vf_logits):
        """
        Computes the next phase fractions using softmax for stability.
        Args:
            current_phase_fractions: Tensor [batch_size, num_particles, num_phases], current VFs.
                                     Used for potential residual connection if desired.
            network_vf_logits: Tensor [batch_size, num_particles, num_phases], raw logits from the network.
        """
        if self.num_phases <= 1:
            return current_phase_fractions

        # Option 1: Network predicts new logits directly
        # For stability, it can be beneficial if the network predicts a *change* to the logits
        # or if the logits are somehow scaled relative to the current state.
        # For now, let's assume network_vf_logits are the new logits.
        
        # Residual connection to logits (optional, can help learning identity for stable regions)
        # One way to add residual: transform current_phase_fractions to a logit-like space
        # inverse_softmax_approx = tf.math.log(current_phase_fractions + 1e-8) # Add epsilon
        # combined_logits = inverse_softmax_approx + network_vf_logits # Network learns a delta in logit space
        
        # Or simpler: network directly predicts the logits for the next step
        combined_logits = network_vf_logits

        # Apply softmax to get normalized, non-negative phase fractions
        next_fractions = tf.nn.softmax(combined_logits, axis=-1)
        
        return next_fractions

    # --- 辅助方法 ---

    def integrate_pos_vel(self, pos1, vel1):
        dt = self.timestep
        vel2 = vel1 + dt * self.gravity
        pos2 = pos1 + dt * (vel1 + vel2) / 2.0 # More stable: use average velocity over timestep
        return pos2, vel2

    def compute_new_pos_vel(self, pos1, vel1, pos2_integrated, vel2_integrated, pos_correction):
        dt = self.timestep
        # Apply correction to the integrated position
        pos_final = pos2_integrated + pos_correction
        # Velocity is based on the change from original position to final corrected position
        vel_final = (pos_final - pos1) / dt
        return pos_final, vel_final

    def compute_position_correction_from_shared(self, shared_features, pos):
        """基于共享特征预测位置修正"""
        filter_extent_tensor = tf.constant(self.filter_extent, dtype=tf.float32)
        
        # 位置专用的最终预测层
        current_features = tf.keras.activations.relu(shared_features)
        
        pos_conv = self.pos_final_conv(current_features, pos, pos, filter_extent_tensor)
        pos_dense = self.pos_final_dense(current_features)
        
        # 合并卷积和全连接的输出
        pos_output = pos_conv + pos_dense
        
        # 缩放位置修正
        pos_correction = (1.0 / 128.0) * pos_output
        
        debug_print("Shape of position correction: ", tf.shape(pos_correction))
        return pos_correction

    def compute_vf_logits_from_shared(self, shared_features, pos):
        """基于共享特征预测相分数logits"""
        if self.num_phases <= 1:
            return None
            
        filter_extent_tensor = tf.constant(self.filter_extent, dtype=tf.float32)
        
        # 相分数专用的最终预测层
        current_features = tf.keras.activations.relu(shared_features)
        
        vf_conv = self.vf_final_conv(current_features, pos, pos, filter_extent_tensor)
        vf_dense = self.vf_final_dense(current_features)
        
        # 合并卷积和全连接的输出
        vf_logits = vf_conv + vf_dense
        
        debug_print("Shape of vf_logits from shared network: ", tf.shape(vf_logits))
        return vf_logits

    def init(self, feats_shape=None):
        """使用虚拟数据初始化模型"""
        pos = np.zeros(shape=(1, 3), dtype=np.float32)
        vel = np.zeros(shape=(1, 3), dtype=np.float32)

        if self.num_phases > 1:
            phase_fractions = np.zeros(shape=(1, self.num_phases), dtype=np.float32)
            phase_fractions[:, 0] = 1.0
        else:
            phase_fractions = None

        box = np.zeros(shape=(1, 3), dtype=np.float32)
        box_feats = np.zeros(shape=(1, 3), dtype=np.float32)

        cd = np.float32(0.5)
        cf = np.float32(0.5)
        
        _ = self.__call__((pos, vel, phase_fractions, box, box_feats), cd=cd, cf=cf)
        
        print(f"{self.name} initialized with {self.num_phases} phases using shared feature extraction.")
        print(f"Shared feature channels: {self.shared_feature_channels}")
        print(f"Position output channels: {self.pos_output_channels}")
        if self.num_phases > 1:
            print(f"VF output channels: {self.vf_output_channels}")
        
        try:
            self.summary()
        except Exception as e:
            print(f"Could not print model summary: {e}")