import tensorflow as tf
import open3d.ml.tf as ml3d
import numpy as np


class VolumeConservationLayer(tf.keras.layers.Layer):
    """确保体积分数修正满足全局守恒的自定义层"""
    
    def __init__(self, num_phases):
        super(VolumeConservationLayer, self).__init__(name="VolumeConservation")
        self.num_phases = num_phases
    
    def call(self, inputs):
        """
        应用体积守恒约束
        Args:
            inputs: 体积分数修正 [batch_size, num_phases-1]
        """
        # 计算每个相的总修正
        batch_size = tf.shape(inputs)[0]
        sum_corrections = tf.reduce_sum(inputs, axis=0, keepdims=True)
        
        # 计算每个粒子应贡献的平均修正（保持总和为零）
        mean_correction = sum_corrections / tf.cast(batch_size, tf.float32)
        
        # 从每个粒子的修正中减去平均值，确保总和为零
        conservation_corrected = inputs - mean_correction
        
        return conservation_corrected


class MultiPhaseParticleNetwork(tf.keras.Model):

    def __init__(self,
                 kernel_size=[4, 4, 4],
                 radius_scale=1.5,
                 coordinate_mapping='ball_to_cube_volume_preserving',
                 interpolation='linear',
                 use_window=True,
                 particle_radius=0.025,
                #  particle_radius=0.05,
                 timestep=1 / 50,
                 gravity=(0, -9.81, 0),
                 num_phases=2,  # 流体相数量
                 cd_cf_as_input=True):  # 是否将cd/cf作为输入
        super().__init__(name=type(self).__name__)
        # 增加输出通道以处理每个相的修正
        self.layer_channels = [32, 64, 64, 3 + num_phases - 1]  # 位置修正 + 体积分数修正(n-1)相
        self.kernel_size = kernel_size
        self.radius_scale = radius_scale
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.use_window = use_window
        self.particle_radius = particle_radius
        self.filter_extent = np.float32(self.radius_scale * 6 *
                                        self.particle_radius)
        self.timestep = timestep
        self.gravity = gravity

        self.num_phases = num_phases
        self.cd_cf_as_input = cd_cf_as_input
        self.initial_phase_volumes = None

        self._all_convs = []

        def window_poly6(r_sqr):
            return tf.clip_by_value((1 - r_sqr)**3, 0, 1)

        def Conv(name, activation=None, **kwargs):
            conv_fn = ml3d.layers.ContinuousConv

            window_fn = None
            if self.use_window == True:
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

        self.conv0_fluid = Conv(name="conv0_fluid",
                                filters=self.layer_channels[0],
                                activation=None)
        self.conv0_obstacle = Conv(name="conv0_obstacle",
                                   filters=self.layer_channels[0],
                                   activation=None)
        self.dense0_fluid = tf.keras.layers.Dense(name="dense0_fluid",
                                                  units=self.layer_channels[0],
                                                  activation=None)

        self.convs = []
        self.denses = []
        for i in range(1, len(self.layer_channels)):
            ch = self.layer_channels[i]
            dense = tf.keras.layers.Dense(units=ch,
                                          name="dense{0}".format(i),
                                          activation=None)
            conv = Conv(name='conv{0}'.format(i), filters=ch, activation=None)
            self.denses.append(dense)
            self.convs.append(conv)
            
        # 添加体积守恒层
        if num_phases > 1:
            self.volume_conservation = VolumeConservationLayer(num_phases-1)
        else:
            self.volume_conservation = None

    def integrate_pos_vel(self, pos1, vel1):
        """应用重力并整合位置和速度"""
        dt = self.timestep
        vel2 = vel1 + dt * tf.constant(self.gravity)
        pos2 = pos1 + dt * (vel2 + vel1) / 2
        return pos2, vel2

    def compute_new_pos_vel(self, pos1, vel1, pos2, vel2, pos_correction):
        """应用位置修正并更新速度"""
        dt = self.timestep
        pos = pos2 + pos_correction
        vel = (pos - pos1) / dt
        return pos, vel
        
    def compute_new_phase_fractions(self, old_fractions, correction, cd, cf):
        """计算更新的体积分数，确保整体体积守恒
        
        Args:
            old_fractions: 原始体积分数 [batch_size, num_phases-1]
            correction: 网络预测的体积分数修正
            cd: 交换系数 (0-1)
            cf: 扩散系数 (0-1)
        """
        # 应用修正
        scaled_correction = correction * cd * cf
        
        # 应用已经通过volume_conservation层处理过的修正
        new_fractions = old_fractions + scaled_correction
        
        # 确保体积分数在合理范围内 (0-1)
        new_fractions = tf.clip_by_value(new_fractions, 0.0, 1.0)
        
        # 归一化处理前后的总体积
        old_sum = tf.reduce_sum(old_fractions, axis=0)
        new_sum = tf.reduce_sum(new_fractions, axis=0)
        
        # 计算缩放因子以保持总体积
        scale_factor = tf.where(new_sum > 0, 
                               old_sum / (new_sum + 1e-6),
                               tf.ones_like(new_sum))
        
        # 应用缩放因子保持每个相的总体积不变
        conserved_fractions = new_fractions * scale_factor
        
        # 确保每个粒子的体积分数总和不超过1（为最后一相留出空间）
        sum_per_particle = tf.reduce_sum(conserved_fractions, axis=-1, keepdims=True)
        scale_norm = tf.where(sum_per_particle > 1.0, 
                            1.0 / (sum_per_particle + 1e-6),
                            tf.ones_like(sum_per_particle))
        
        return conserved_fractions * scale_norm

    def compute_correction(self,
                           pos,
                           vel,
                           phase_fractions,
                           box,
                           box_feats,
                           fixed_radius_search_hash_table=None,
                           cd=0.5,
                           cf=0.5):
        """计算位置和体积分数的修正
        
        Args:
            pos: 粒子位置
            vel: 粒子速度
            phase_fractions: 体积分数 [batch_size, num_phases-1]
            box: 障碍物位置
            box_feats: 障碍物特征
            fixed_radius_search_hash_table: 可选的哈希表加速搜索
            cd: 交换系数
            cf: 扩散系数
        """
        # 计算过滤器范围
        filter_extent = tf.constant(self.filter_extent)

        # 确保cd和cf是标量
        cd = tf.convert_to_tensor(cd, dtype=tf.float32)
        cf = tf.convert_to_tensor(cf, dtype=tf.float32)
        
        # 构建基础特征：位置标识符和速度
        fluid_feats = [tf.ones_like(pos[:, 0:1]), vel]
        
        # 添加体积分数信息
        if (phase_fractions is not None):
            fluid_feats.append(phase_fractions)
        
        # 可选：将cd和cf作为全局特征
        if self.cd_cf_as_input:
            # 确保创建的张量形状匹配
            cd_scalar = tf.reshape(cd, ())  # 将cd转换为标量
            cf_scalar = tf.reshape(cf, ())  # 将cf转换为标量
            
            cd_tensor = tf.ones_like(pos[:, 0:1]) * cd_scalar
            cf_tensor = tf.ones_like(pos[:, 0:1]) * cf_scalar
            fluid_feats.extend([cd_tensor, cf_tensor])
        
        # 合并所有特征
        fluid_feats = tf.concat(fluid_feats, axis=-1)

        # 网络前向传播
        self.ans_conv0_fluid = self.conv0_fluid(fluid_feats, pos, pos,
                                                filter_extent)
        self.ans_dense0_fluid = self.dense0_fluid(fluid_feats)
        self.ans_conv0_obstacle = self.conv0_obstacle(box_feats, box, pos,
                                                      filter_extent)

        feats = tf.concat([
            self.ans_conv0_obstacle, self.ans_conv0_fluid, self.ans_dense0_fluid
        ],
                          axis=-1)

        self.ans_convs = [feats]
        for conv, dense in zip(self.convs, self.denses):
            inp_feats = tf.keras.activations.relu(self.ans_convs[-1])
            ans_conv = conv(inp_feats, pos, pos, filter_extent)
            ans_dense = dense(inp_feats)
            if ans_dense.shape[-1] == self.ans_convs[-1].shape[-1]:
                ans = ans_conv + ans_dense + self.ans_convs[-1]
            else:
                ans = ans_conv + ans_dense
            self.ans_convs.append(ans)

        # 计算流体邻居数量（用于训练中的损失函数）
        self.num_fluid_neighbors = ml3d.ops.reduce_subarrays_sum(
            tf.ones_like(self.conv0_fluid.nns.neighbors_index,
                         dtype=tf.float32),
            self.conv0_fluid.nns.neighbors_row_splits)

        self.last_features = self.ans_convs[-2]

        # 缩放以更好地匹配输出分布
        raw_correction = (1.0 / 128) * self.ans_convs[-1]
        
        # 分离位置和体积分数修正
        pos_correction = raw_correction[:, :3]
        
        # 如果是多相流体，处理体积分数修正
        if self.num_phases > 1 and phase_fractions is not None:
            vol_frac_correction = raw_correction[:, 3:]
            
            # 应用体积守恒约束
            if self.volume_conservation is not None:
                vol_frac_correction = self.volume_conservation(vol_frac_correction)
            
            # 存储修正
            self.pos_correction = pos_correction
            self.vol_frac_correction = vol_frac_correction
            
            # 合并为完整的修正向量
            correction = tf.concat([pos_correction, vol_frac_correction], axis=-1)
            return correction
        else:
            self.pos_correction = pos_correction
            self.vol_frac_correction = None
            return pos_correction

    def set_initial_phase_volumes(self, phase_fractions):
        """设置初始相体积值作为参考"""
        if self.num_phases > 1 and phase_fractions is not None:
            # 使用变量而非直接赋值，以确保图兼容性
            if self.initial_phase_volumes is None:
                # 计算每个相的总体积
                total_volumes = tf.reduce_sum(phase_fractions, axis=0)
                # 初始化为变量
                self.initial_phase_volumes = tf.Variable(
                    total_volumes, 
                    trainable=False, 
                    name="initial_phase_volumes")
            else:
                # 如果已经初始化，则更新变量
                self.initial_phase_volumes.assign(tf.reduce_sum(phase_fractions, axis=0))
            
            # 使用print而非tf.print，避免图模式问题
            print("Initial phase volumes set:", self.initial_phase_volumes.numpy())

    def enforce_global_conservation(self, phase_fractions):
        """强制执行全局体积守恒"""
        if self.initial_phase_volumes is None or self.num_phases <= 1 or phase_fractions is None:
            return phase_fractions
        
        # 确保使用tf.Variable的值，兼容图模式
        init_volumes = self.initial_phase_volumes
            
        # 计算当前总体积
        current_volumes = tf.reduce_sum(phase_fractions, axis=0)
        
        # 计算全局缩放因子
        scaling = init_volumes / (current_volumes + 1e-6)
        
        # 应用缩放以维持全局体积
        conserved_fractions = phase_fractions * scaling
        
        return conserved_fractions

    def normalize_phase_fractions(self, fractions):
        """确保每个粒子的体积分数有效且保持总体积守恒"""
        if fractions is None or self.num_phases <= 1:
            return fractions
            
        # 裁剪负值
        fractions_clipped = tf.maximum(fractions, 0.0)
        
        # 计算裁剪前后的总体积变化
        total_before = tf.reduce_sum(fractions, axis=0)
        total_after = tf.reduce_sum(fractions_clipped, axis=0)
        
        # 计算补偿因子以保持总体积
        scale = tf.where(total_after > 0, 
                        total_before / (total_after + 1e-6), 
                        tf.ones_like(total_after))
        
        # 应用补偿因子
        fractions_conserved = fractions_clipped * scale
        
        # 确保每个粒子的体积分数总和不超过1
        sum_per_particle = tf.reduce_sum(fractions_conserved, axis=-1, keepdims=True)
        scale_factor = tf.where(sum_per_particle > 1.0, 
                              1.0 / (sum_per_particle + 1e-6),
                              tf.ones_like(sum_per_particle))
        
        fractions_normalized = fractions_conserved * scale_factor
        
        # 再次检查全局守恒并调整
        total_final = tf.reduce_sum(fractions_normalized, axis=0)
        final_scale = total_before / (total_final + 1e-6)
        
        return fractions_normalized * final_scale

    def call(self, inputs, fixed_radius_search_hash_table=None, cd=0.5, cf=0.5):
        """计算多相流体的一个时间步"""
        pos, vel, phase_fractions, box, box_feats = inputs
        
        # 在eager模式下设置初始体积分布
        if not tf.executing_eagerly() and self.initial_phase_volumes is None and phase_fractions is not None:
            # 在图模式下，我们需要另外处理
            # 创建一个变量但暂不赋值，会在第一次执行时赋值
            self.initial_phase_volumes = tf.Variable(
                tf.zeros_like(tf.reduce_sum(phase_fractions, axis=0)),
                trainable=False,
                name="initial_phase_volumes")
        elif tf.executing_eagerly() and self.initial_phase_volumes is None and phase_fractions is not None:
            # 仅在eager模式下设置
            self.set_initial_phase_volumes(phase_fractions)
            
        # 整合位置和速度
        pos2, vel2 = self.integrate_pos_vel(pos, vel)
        
        # 计算修正
        corrections = self.compute_correction(
            pos2, vel2, phase_fractions, box, box_feats, 
            fixed_radius_search_hash_table, cd, cf)
        
        # 分解修正
        if self.num_phases > 1 and phase_fractions is not None:
            pos_correction = corrections[:, :3]
            vol_frac_correction = corrections[:, 3:]
        else:
            pos_correction = corrections
            vol_frac_correction = None
        
        # 更新位置和速度
        pos2_corrected, vel2_corrected = self.compute_new_pos_vel(
            pos, vel, pos2, vel2, pos_correction)
        
        # 更新体积分数
        if phase_fractions is not None and vol_frac_correction is not None:
            phase_fractions_corrected = self.compute_new_phase_fractions(
                phase_fractions, vol_frac_correction, cd, cf)
            
            # 应用全局体积守恒
            phase_fractions_corrected = self.enforce_global_conservation(
                phase_fractions_corrected)
                
            # 确保每个粒子的体积分数有效
            phase_fractions_corrected = self.normalize_phase_fractions(
                phase_fractions_corrected)
                
            return pos2_corrected, vel2_corrected, phase_fractions_corrected
        else:
            return pos2_corrected, vel2_corrected

    def init(self, feats_shape=None):
        """使用虚拟数据初始化所有变量的形状"""
        pos = np.zeros(shape=(1, 3), dtype=np.float32)
        vel = np.zeros(shape=(1, 3), dtype=np.float32)
        
        # 为多相流添加体积分数
        if self.num_phases > 1:
            # 为n-1个相创建体积分数（最后一个相可以计算得出）
            phase_fractions = np.zeros(shape=(1, self.num_phases-1), dtype=np.float32)
            # 初始设置第一相的分数为1
            phase_fractions[:, 0] = 1.0
        else:
            phase_fractions = None
            
        box = np.zeros(shape=(1, 3), dtype=np.float32)
        box_feats = np.zeros(shape=(1, 3), dtype=np.float32)

        # 添加cd和cf参数以确保特征维度匹配
        cd = np.float32(0.5)
        cf = np.float32(0.5)

        # 调用模型确保变量初始化
        _ = self.__call__((pos, vel, phase_fractions, box, box_feats), cd=cd, cf=cf)
        
        print("Network initialized with", self.num_phases, "phases")
