import tensorflow as tf
import open3d.ml.tf as ml3d
import numpy as np
from debug_utils import debug_print
from utils.window_func import get_window_func
from utils.convolutions import ContinuousConv
from utils.multi_density_continuous_conv_2 import MultiDensityContinuousConv


class MyParticleNetwork(tf.keras.Model):

    def __init__(self,
                 kernel_size=[4, 4, 4],
                 sym_kernel_size=[6, 6, 6],
                 sym_axis=2,
                 radius_scale=1.5,
                 box_radius_scale=1.5,
                 coordinate_mapping='ball_to_cube_volume_preserving',
                 interpolation='linear',
                 use_window=True,
                 window="poly6",
                 window_custom="custom_density_window",
                 use_sym=True,    # 只针对最后一层
                 window_sym="peak",
                 window_custom_sym="custom_density_window_sym",
                 ignore_query_points=False,
                 particle_radius=0.025,
                 timestep=1 / 50,
                 gravity=(0, -9.81, 0),
                 multi_density_support=True,  # 是否支持多密度流体
                 density_embed_dim=8,        # 密度嵌入维度
                 density_condition_layers=True,  # 是否对卷积层进行密度条件化
                 density_relative=True,       # 是否使用相对密度(相对于rest_dens)
                 use_density=True,            # 是否使用密度特征
                 density_embed=False,          # 是否对密度特征进行embedding
                 use_fluid_ones=True,          # 是否使用占位符 1 作为流体粒子特征
                 use_all_dens_condtition=False,
                 rest_dens=1000,
                 circular=False,
                 # 🔥 新增: 密度条件化模式选择
                 density_modulation_mode='density_ratio',  # 'density_ratio' 或 'pairwise_film'
                 film_hidden_dim=16):
        super().__init__(name=type(self).__name__)
        self.layer_channels = [32, 64, 64, 3]
        self.kernel_size = kernel_size
        self.sym_kernel_size = sym_kernel_size
        self.sym_axis = sym_axis
        self.radius_scale = radius_scale
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.use_window = use_window
        self.particle_radius = particle_radius
        self.filter_extent = np.float32(self.radius_scale * 6 *
                                        self.particle_radius)
        self.timestep = timestep
        self.gravity = gravity
        self.circular = circular
        self.ignore_query_points = ignore_query_points

        self.window = window
        self.window_custom = window_custom

        self.use_sym = use_sym
        self.window_sym = window_sym
        self.window_custom_sym = window_custom_sym

        self.use_fluid_ones = use_fluid_ones
        # 多密度流体相关参数
        self.multi_density_support = multi_density_support
        self.density_embed_dim = density_embed_dim
        self.density_condition_layers = density_condition_layers
        self.density_relative = density_relative
        self.use_density = use_density
        self.density_embed = density_embed
        self.rest_dens = rest_dens
        self.use_all_dens_condtition = use_all_dens_condtition

        # 🔥 新增: 存储密度条件化模式参数
        self.density_modulation_mode = density_modulation_mode
        self.film_hidden_dim = film_hidden_dim

        debug_print(f"particle_radius: {self.particle_radius}")
        debug_print(f"filter_extent: {self.filter_extent}")
        debug_print(f"use_all_dens_condtition: {self.use_all_dens_condtition}")

        self._all_convs = []

        if self.multi_density_support and self.density_embed and self.use_density:
            self.density_embedding = tf.keras.Sequential([
                tf.keras.layers.Dense(density_embed_dim, activation='relu'),
                tf.keras.layers.Dense(density_embed_dim)
            ], name='density_embed')

        self.conv0_fluid = self.get_cconv(name='conv0_fluid',
                                          filters=self.layer_channels[0],
                                          activation=None,
                                          window_func=self.window,
                                          custom_window_func=self.window_custom,
                                          force_density_condition=True,
                                          circular=circular)
        self.conv0_obstacle = self.get_cconv(name='conv0_obstacle',
                                             filters=self.layer_channels[0],
                                             activation=None,
                                             window_func=self.window,
                                             custom_window_func=self.window_custom,
                                            #  force_density_condition=True,
                                             circular=circular)
        self.dense0_fluid = tf.keras.layers.Dense(name="dense0_fluid",
                                                  units=self.layer_channels[0],
                                                  activation=None)
        # self.conv0_fluid = Conv(name="conv0_fluid",
        #                         filters=self.layer_channels[0],
        #                         activation=None)
        # self.conv0_obstacle = Conv(name="conv0_obstacle",
        #                            filters=self.layer_channels[0],
        #                            activation=None)
        # self.dense0_fluid = tf.keras.layers.Dense(name="dense0_fluid",
        #                                           units=self.layer_channels[0],
        #                                           activation=None)

        self.convs = []
        self.denses = []
        for i in range(1, len(self.layer_channels) - 1):
            ch = self.layer_channels[i]
            dense = tf.keras.layers.Dense(units=ch,
                                          name="dense{0}".format(i),
                                          activation=None)
            if self.use_all_dens_condtition:
                conv = self.get_cconv(name='conv{0}'.format(i),
                                    filters=ch,
                                    activation=None,
                                    window_func=self.window,
                                    custom_window_func=self.window_custom,
                                    ignore_query_points=self.ignore_query_points,
                                    force_density_condition=True,
                                    circular=self.circular)
            else:
                conv = self.get_cconv(name='conv{0}'.format(i),
                                  filters=ch,
                                  activation=None,
                                  window_func=self.window,
                                  custom_window_func=self.window_custom,
                                  ignore_query_points=self.ignore_query_points,
                                  circular=self.circular)
            self.denses.append(dense)
            self.convs.append(conv)

        ch = self.layer_channels[-1]

        debug_print("use_sym: ", self.use_sym)
        if self.use_sym:
            debug_print("sym_axis: ", self.sym_axis)
            sym_conv = self.get_cconv(name=f'sym_conv',
                                      filters=ch,
                                      activation=None,
                                      use_bias=False,
                                      symmetric=True,
                                      kernel_size=self.sym_kernel_size,
                                      ignore_query_points=True,
                                      window_func=self.window_sym or self.window,
                                      custom_window_func=self.window_custom_sym,
                                      sym_axis=self.sym_axis,
                                      circular=self.circular)
        else:
            sym_conv = self.get_cconv(name='conv3',
                                    filters=ch,
                                    activation=None,
                                    window_func=self.window,
                                    custom_window_func=self.window_custom,
                                    force_density_condition=True,
                                    circular=circular)
        self.sym_conv = sym_conv

    def get_cconv(self,
                  name,
                  filters,
                  kernel_size=None,
                  activation=None,
                  ignore_query_points=None,
                  window_func=None,
                  normalize=False,
                  **kwargs):

        if kernel_size is None:
            kernel_size = self.kernel_size
        if ignore_query_points is None:
            ignore_query_points = self.ignore_query_points

        # 使用混合类中的方法选择合适的卷积类型
        conv = self.choose_conv_type(
            name=name,
            kernel_size=kernel_size,
            filters=filters,
            activation=activation,
            window_func=get_window_func(window_func),
            ignore_query_points=ignore_query_points,
            normalize=normalize,
            interpolation=self.interpolation,
            coordinate_mapping=self.coordinate_mapping,
            **kwargs
        )

        self._all_convs.append((name, conv))
        return conv

    def integrate_pos_vel(self, pos1, vel1):
        """Apply gravity and integrate position and velocity"""
        dt = self.timestep
        vel2 = vel1 + dt * tf.constant(self.gravity)
        pos2 = pos1 + dt * (vel2 + vel1) / 2
        return pos2, vel2

    def compute_new_pos_vel(self, pos1, vel1, pos2, vel2, pos_correction):
        """Apply the correction
        pos1,vel1 are the positions and velocities from the previous timestep
        pos2,vel2 are the positions after applying gravity and the integration step
        """
        dt = self.timestep
        pos = pos2 + pos_correction
        vel = (pos - pos1) / dt
        return pos, vel

    def compute_correction(self,
                           pos,
                           vel,
                           density,
                           other_feats,
                           box,
                           box_feats,
                           fixed_radius_search_hash_table=None):
        """Expects that the pos and vel has already been updated with gravity and velocity"""
        debug_print("compute_correction")
        debug_print("pos.shape: ", pos.shape)
        debug_print("vel.shape: ", vel.shape)
        debug_print("density.shape: ", density.shape)
        debug_print("pos 300: ", pos[::300])
        debug_print("vel 300: ", vel[::300])
        debug_print("density 300: ", density[::300])
        # debug_print("box.shape: ", box.shape)
        # debug_print("box_feats.shape: ", box_feats.shape)
        # debug_print("box 10: ", box[0:10])
        # debug_print("box_feats 10: ", box_feats[0:10])

        # compute the extent of the filters (the diameter)
        filter_extent = tf.constant(self.filter_extent)

        if self.use_fluid_ones:
            fluid_feats = [tf.ones_like(pos[:, 0:1]), vel]
        else:
            fluid_feats = [vel]
        if not other_feats is None:
            fluid_feats.append(other_feats)
        if self.multi_density_support and self.use_density:
            if self.density_embed:
                debug_print("self.density_embed: ", self.density_embed)
                density_feat = self.density_embedding(density)
                debug_print("density_feat 300: ", density_feat[::300])
                fluid_feats.append(density_feat)
            elif density is not None:
                fluid_feats.append(density)
        fluid_feats = tf.concat(fluid_feats, axis=-1)
        debug_print("fluid_feats.shape: ", fluid_feats.shape)

        if type(self.conv0_fluid).__name__ == 'MultiDensityContinuousConv':
            self.ans_conv0_fluid = self.conv0_fluid(fluid_feats, pos, pos,
                                                    filter_extent, density, density)
        else:
            self.ans_conv0_fluid = self.conv0_fluid(fluid_feats, pos, pos,
                                                    filter_extent)
        self.ans_dense0_fluid = self.dense0_fluid(fluid_feats)
        
        if type(self.conv0_obstacle).__name__ == 'MultiDensityContinuousConv':
            num_box_particles = tf.shape(box)[0]
            # 为边界粒子创建一个代表其"密度"的张量
            # 你需要决定边界的密度表示。这里假设使用一个常数。
            # 如果 density_relative=True (使用相对密度)，则使用 1.0 或一个更大的相对值
            # 如果 density_relative=False (使用绝对密度)，则使用 self.rest_dens 或一个更大的绝对值
            # 更好的做法是使用一个与训练数据中流体密度范围相当的、或者更大的常数
            # 例如，如果相对密度范围是 [0.5, 10]，边界相对密度可以设为 1.0 或 5.0
            # 如果绝对密度范围是 [500, 10000]，边界绝对密度可以设为 1000 或 5000
            boundary_constant_density_value = 10.0 if self.density_relative else self.rest_dens # 选择一个合适的值
            boundary_densities = tf.fill([num_box_particles, 1], boundary_constant_density_value) # 形状 [M, 1]
            # 输入点是 box，输入特征是 box_feats，输入密度是 boundary_densities
            # 输出点是 pos，输出密度是 density
            self.ans_conv0_obstacle = self.conv0_obstacle(
                box_feats,
                box, # 输入点位置
                pos, # 输出点位置 (查询点)
                filter_extent,
                inp_densities=boundary_densities, # 输入点的密度 (边界密度)
                out_densities=density, # 输出点的密度 (流体密度)
            )
        else:
             # 这个应该被执行，因为obstacle 是一个 ContinuousConv 层
            self.ans_conv0_obstacle = self.conv0_obstacle(
                box_feats,
                box,
                pos,
                filter_extent,
                # 注意：如果这里是 ContinuousConv，这些密度参数是不需要的
                # inp_densities=None, # 或其他表示没有密度的方式
                # out_densities=None,
            )

        feats = tf.concat([
            self.ans_conv0_obstacle, self.ans_conv0_fluid, self.ans_dense0_fluid
        ],
            axis=-1)

        self.ans_convs = [feats]
        for conv, dense in zip(self.convs, self.denses):
            inp_feats = tf.keras.activations.relu(self.ans_convs[-1])
            if type(conv).__name__ == 'MultiDensityContinuousConv':
                ans_conv = conv(inp_feats, pos, pos, filter_extent, density,
                                density)
            else:
                ans_conv = conv(inp_feats, pos, pos, filter_extent)
            ans_dense = dense(inp_feats)
            if ans_dense.shape[-1] == self.ans_convs[-1].shape[-1]:
                ans = ans_conv + ans_dense + self.ans_convs[-1]
            else:
                ans = ans_conv + ans_dense
            self.ans_convs.append(ans)

        # compute the number of fluid neighbors.
        # this info is used in the loss function during training.
        self.num_fluid_neighbors = ml3d.ops.reduce_subarrays_sum(
            tf.ones_like(self.conv0_fluid.nns.neighbors_index,
                         dtype=tf.float32),
            self.conv0_fluid.nns.neighbors_row_splits)
        
        num_box_neighbors = ml3d.ops.reduce_subarrays_sum(
            tf.ones_like(self.conv0_obstacle.nns.neighbors_index,
                         dtype=tf.float32),
            self.conv0_obstacle.nns.neighbors_row_splits)        
        debug_print("self.num_fluid_neighbors step 100: ", self.num_fluid_neighbors[::100])
        debug_print("self.num_box_neighbors step 100: ", num_box_neighbors[::100])

        # self.last_features = self.ans_convs[-2]
        self.last_features = self.ans_convs[-1]
        _feats = tf.keras.activations.relu(self.ans_convs[-1])

        # 处理最后一层对称卷积
        if type(self.sym_conv).__name__ == 'MultiDensityContinuousConv':
            final_ans = self.sym_conv(_feats, pos, pos,
                                      filter_extent, density, density)
        else:
            final_ans = self.sym_conv(_feats, pos, pos,
                                      filter_extent, None)
            # scale to better match the scale of the output distribution
        # self.pos_correction = (1.0 / 128) * self.ans_convs[-1]
        self.pos_correction = (1.0 / 128) * final_ans
        debug_print("self.pos_correction 10: ", self.pos_correction[0:10])
        debug_print("self.pos_correction 300: ", self.pos_correction[::300])
        return self.pos_correction

    def call(self, inputs, fixed_radius_search_hash_table=None):
        """computes 1 simulation timestep
        inputs: list or tuple with (pos,vel,feats,box,box_feats)
          pos and vel are the positions and velocities of the fluid particles.
          feats is reserved for passing additional features, use None here.
          box are the positions of the static particles and box_feats are the
          normals of the static particles.
        """
        pos, vel, density, feats, box, box_feats = inputs
        if self.density_relative:
            density = density / self.rest_dens
        density = tf.expand_dims(density, -1)

        pos2, vel2 = self.integrate_pos_vel(pos, vel)
        pos_correction = self.compute_correction(
            pos2, vel2, density, feats, box, box_feats, fixed_radius_search_hash_table)
        pos2_corrected, vel2_corrected = self.compute_new_pos_vel(
            pos, vel, pos2, vel2, pos_correction)

        return pos2_corrected, vel2_corrected

    def init(self, feats_shape=None):
        """Runs the network with dummy data to initialize the shape of all variables"""
        pos = np.zeros(shape=(1, 3), dtype=np.float32)
        vel = np.zeros(shape=(1, 3), dtype=np.float32)
        density = np.zeros(shape=(1), dtype=np.float32)
        if feats_shape is None:
            feats = None
        else:
            feats = np.zeros(shape=feats_shape, dtype=np.float32)
        box = np.zeros(shape=(1, 3), dtype=np.float32)
        box_feats = np.zeros(shape=(1, 3), dtype=np.float32)

        _ = self.__call__((pos, vel, density, feats, box, box_feats))

    def choose_conv_type(self,
                         name,
                         kernel_size,
                         filters,
                         activation=None,
                         window_func=None,
                         ignore_query_points=False,
                         normalize=False,
                         interpolation='linear',
                         coordinate_mapping='ball_to_cube_volume_preserving',
                         circular=False,
                         **kwargs):
        """智能选择使用普通卷积还是密度条件卷积"""

        # 根据层的用途决定是否使用密度条件化卷积
        use_density_conditioning = hasattr(
            self, 'density_condition_layers') and self.density_condition_layers

        # 从层名称判断是否需要密度条件化
        # 一些关键层应始终使用密度条件化卷积（如流体和主要卷积层）
        is_critical_layer = ('final' in name or
                             'sym' in name)    # 最终输出层
        # 从kwargs中提取force_density_condition参数，不再向下传递
        force_density_condition = kwargs.pop(
            'force_density_condition', False)
        custom_window_func = kwargs.pop('custom_window_func', None)

        if use_density_conditioning and (is_critical_layer or force_density_condition):

            return MultiDensityContinuousConv(
                name=name,
                kernel_size=kernel_size,
                filters=filters,
                activation=activation,
                align_corners=True,
                interpolation=interpolation,
                coordinate_mapping=coordinate_mapping,
                normalize=normalize,
                window_function=None,
                combined_importance_function=get_window_func(
                    custom_window_func),       # 使用自定义窗口函数
                radius_search_ignore_query_points=ignore_query_points,
                use_dense_layer_for_center=False,
                circular=circular,
                # 🔥 新增: 传递密度条件化模式参数
                density_modulation_mode=self.density_modulation_mode,
                film_hidden_dim=self.film_hidden_dim,
                **kwargs
            )
        else:
            return ContinuousConv(
                name=name,
                kernel_size=kernel_size,
                filters=filters,
                activation=activation,
                align_corners=True,
                interpolation=interpolation,
                coordinate_mapping=coordinate_mapping,
                normalize=normalize,
                window_function=window_func,
                radius_search_ignore_query_points=ignore_query_points,
                use_dense_layer_for_center=False,
                circular=circular,
                **kwargs
            )
