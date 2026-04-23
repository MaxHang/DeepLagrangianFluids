"""
多密度连续卷积层 (Multi-Density Continuous Convolution Layer)

该模块实现了用于处理多密度流体的自定义连续卷积层，支持两种密度条件化模式: 
1. density_ratio: 简单的密度比权重 (ρⱼ/ρᵢ)
2. pairwise_film: Pair-wise FiLM调制 (γᵢⱼ^ρ ⊙ fⱼ + βᵢⱼ^ρ)

作者: [Your Name]
日期: 2025-11-05
"""

import tensorflow as tf
try:
    from utils.convolutions import ContinuousConv
except ModuleNotFoundError:
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from convolutions import ContinuousConv
import open3d.ml.tf as ml3d
import numpy as np
from typing import Optional, Callable, Tuple


class MultiDensityContinuousConv(ContinuousConv):
    """
    多密度连续卷积层
    
    继承自 ContinuousConv，扩展支持基于密度的条件化机制。
    支持两种模式: 
    1. 'density_ratio': w = (ρⱼ/ρᵢ) x spatial_weight
    2. 'pairwise_film': Γᵢⱼ^ρ = γᵢⱼ^ρ ⊙ fⱼ + βᵢⱼ^ρ
    
    Attributes:
        _density_window_function: 自定义密度窗口函数
        density_modulation_mode: 密度条件化模式 ('density_ratio' 或 'pairwise_film')
        film_hidden_dim: FiLM 网络隐藏层维度
        film_gamma_net: γ 参数生成网络 (仅 pairwise_film 模式)
        film_beta_net: β 参数生成网络 (仅 pairwise_film 模式)
    """
    
    def __init__(self,
                 filters: int,
                 kernel_size: list,
                 activation: Optional[str] = None,
                 use_bias: bool = True,
                 kernel_initializer: str = 'uniform',
                 bias_initializer: str = 'zeros',
                 kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
                 bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
                 align_corners: bool = True,
                 coordinate_mapping: str = 'ball_to_cube_radial',
                 interpolation: str = 'linear',
                 normalize: bool = True,
                 radius_search_ignore_query_points: bool = False,
                 radius_search_metric: str = 'L2',
                 offset: Optional[tf.Tensor] = None,
                 window_function: Optional[Callable] = None,
                 combined_importance_function: Optional[Callable] = None,
                 use_dense_layer_for_center: bool = False,
                 dense_kernel_initializer: str = 'glorot_uniform',
                 dense_kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
                 symmetric: bool = False,
                 sym_axis: int = 2,
                 circular: bool = False,
                 density_modulation_mode: str = 'density_ratio',
                 film_hidden_dim: int = 16,
                 **kwargs):
        """
        初始化多密度连续卷积层
        
        Args:
            filters: 输出特征通道数
            kernel_size: 卷积核空间分辨率，例如 [4, 4, 4]
            activation: 激活函数名称
            use_bias: 是否使用偏置项
            kernel_initializer: 卷积核权重初始化器
            bias_initializer: 偏置项初始化器
            kernel_regularizer: 卷积核权重正则化器
            bias_regularizer: 偏置项正则化器
            align_corners: 坐标映射时是否对齐角点
            coordinate_mapping: 坐标映射方式
            interpolation: 插值方式 ('linear' 或 'nearest_neighbor')
            normalize: 是否进行归一化
            radius_search_ignore_query_points: 半径搜索时是否忽略查询点自身
            radius_search_metric: 半径搜索的距离度量标准 ('L2', 'L1', 'Linf')
            offset: 偏移量
            window_function: 标准窗口函数（通常设为 None）
            combined_importance_function: 自定义重要性函数（密度+空间）
            use_dense_layer_for_center: 是否使用密集层处理中心点
            dense_kernel_initializer: 密集层权重初始化器
            dense_kernel_regularizer: 密集层权重正则化器
            symmetric: 是否使用对称卷积
            sym_axis: 对称轴索引
            circular: 是否使用循环边界条件
            density_modulation_mode: 密度条件化模式
                - 'density_ratio': 简单密度比权重 (默认)
                - 'pairwise_film': Pair-wise FiLM 调制
            film_hidden_dim: FiLM 网络隐藏层维度 (仅 pairwise_film 模式使用)
            **kwargs: 其他参数
        """
        # 调用父类构造函数
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            align_corners=align_corners,
            coordinate_mapping=coordinate_mapping,
            interpolation=interpolation,
            normalize=normalize,
            radius_search_ignore_query_points=radius_search_ignore_query_points,
            radius_search_metric=radius_search_metric,
            offset=offset,
            window_function=combined_importance_function,
            use_dense_layer_for_center=use_dense_layer_for_center,
            dense_kernel_initializer=dense_kernel_initializer,
            dense_kernel_regularizer=dense_kernel_regularizer,
            symmetric=symmetric,
            sym_axis=sym_axis,
            circular=circular,
            **kwargs
        )
        
        # 存储自定义密度窗口函数
        # 存储自定义密度窗口函数
        # 🔥 根据模式选择合适的窗口函数
        if density_modulation_mode == 'pairwise_film':
            # Pair-wise FiLM 模式: 只使用空间权重
            # 因为密度调制已通过 γᵢⱼ^ρ 和 βᵢⱼ^ρ 作用在特征上
            self._density_window_function = lambda d, n, q: tf.clip_by_value(
                (1 - d)**3, 0, 1
            )
            print(f"{self.name} use pairwise_film model: spatial_weight only")
        else:
            # Density ratio 模式: 使用完整的密度+空间窗口函数
            self._density_window_function = combined_importance_function
            print(f"{self.name} use density_ratio model: spatial_weight x density_weight")
        
        # 密度条件化模式配置
        self.density_modulation_mode = density_modulation_mode
        self.film_hidden_dim = film_hidden_dim
        
        # FiLM 网络（延迟初始化，在 build 时创建）
        self.film_gamma_net: Optional[tf.keras.Sequential] = None
        self.film_beta_net: Optional[tf.keras.Sequential] = None
        self._film_nets_built = False
        
        # 验证模式参数
        if density_modulation_mode not in ['density_ratio', 'pairwise_film']:
            raise ValueError(
                f"density_modulation_mode must be 'density_ratio' or 'pairwise_film', "
                f"bug get '{density_modulation_mode}'"
            )

    def build(self, inp_features_shape: tf.TensorShape) -> None:
        """
        构建层，创建权重和 FiLM 网络
        
        Args:
            inp_features_shape: 输入特征的形状 [num_input_points, in_channels]
        """
        # 调用父类 build
        super().build(inp_features_shape)
        
        # 如果使用 pairwise_film 模式且未构建，则创建 FiLM 网络
        if self.density_modulation_mode == 'pairwise_film' and not self._film_nets_built:
            in_channels = inp_features_shape[-1]
            
            # γᵢⱼ^ρ 生成网络: 密度比 → 缩放参数
            # 输入: [num_pairs, 1] → 输出: [num_pairs, in_channels]
            self.film_gamma_net = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    self.film_hidden_dim,
                    activation='relu',
                    name=f'{self.name}_film_gamma_hidden'
                ),
                tf.keras.layers.Dense(
                    in_channels,
                    activation=None,
                    name=f'{self.name}_film_gamma_out'
                )
            ], name=f'{self.name}_film_gamma')
            
            # βᵢⱼ^ρ 生成网络: 密度比 → 平移参数
            # 输入: [num_pairs, 1] → 输出: [num_pairs, in_channels]
            self.film_beta_net = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    self.film_hidden_dim,
                    activation='relu',
                    name=f'{self.name}_film_beta_hidden'
                ),
                tf.keras.layers.Dense(
                    in_channels,
                    activation=None,
                    name=f'{self.name}_film_beta_out'
                )
            ], name=f'{self.name}_film_beta')
            
            self._film_nets_built = True
            
            print(f"[{self.name}] has build Pair-wise FiLM network:")
            print(f"  - input shape: {in_channels}")
            print(f"  - hideen shape: {self.film_hidden_dim}")
            print(f"  - gama/beta output shape: {in_channels}")

    def call(self,
             inp_features: tf.Tensor,
             inp_positions: tf.Tensor,
             out_positions: tf.Tensor,
             extents: tf.Tensor,
             inp_densities: tf.Tensor,
             out_densities: tf.Tensor,
             inp_importance: Optional[tf.Tensor] = None,
             fixed_radius_search_hash_table: Optional[object] = None,
             user_neighbors_index: Optional[tf.Tensor] = None,
             user_neighbors_row_splits: Optional[tf.Tensor] = None,
             user_neighbors_importance: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        前向传播
        
        Args:
            inp_features: 输入特征
                形状: [num_input_points, in_channels]
            inp_positions: 输入点坐标
                形状: [num_input_points, 3]
            out_positions: 输出点坐标（查询点）
                形状: [num_output_points, 3]
            extents: 邻域搜索半径（直径）
                形状: [] (标量) 或 [num_output_points]
            inp_densities: 输入点密度
                形状: [num_input_points, 1]
            out_densities: 输出点密度
                形状: [num_output_points, 1]
                注意: 对于流体自交互，inp_densities == out_densities
            inp_importance: 可选的输入点重要性
                形状: [num_input_points] 或 None
            fixed_radius_search_hash_table: 预计算的哈希表（用于加速）
            user_neighbors_index: 不应提供（本层内部计算）
            user_neighbors_row_splits: 不应提供（本层内部计算）
            user_neighbors_importance: 不应提供（本层内部计算）
        
        Returns:
            输出特征: [num_output_points, filters]
        
        Raises:
            ValueError: 如果提供了 user_neighbors_* 参数
        """
        # 确保 user_neighbors_* 没有被预先提供
        if (user_neighbors_index is not None or 
            user_neighbors_row_splits is not None or 
            user_neighbors_importance is not None):
            raise ValueError(
                "MultiDensityContinuousConv use internal neighbors search, "
                "don't offer user_neighbors_* params"
            )

        # ==================== 步骤 1: 执行邻域搜索 ====================
        return_distances = True
        
        if extents.shape.rank == 0:
            # 所有输出点使用相同的范围
            radius = 0.5 * extents  # 半径 = 直径 / 2
            self.nns = self.fixed_radius_search(
                inp_positions,
                queries=out_positions,
                radius=radius,
                hash_table=fixed_radius_search_hash_table
            )
            
            # 归一化距离（用于窗口函数）
            if return_distances and self.radius_search_metric == 'L2':
                # L2: 归一化为 r²/R²
                neighbors_distance_normalized = self.nns.neighbors_distance / (radius * radius)
            elif return_distances and self.radius_search_metric == 'L1':
                # L1: 归一化为 r/R
                neighbors_distance_normalized = self.nns.neighbors_distance / radius
            elif return_distances:
                # Linf: 不归一化
                neighbors_distance_normalized = self.nns.neighbors_distance
            else:
                neighbors_distance_normalized = None

        elif extents.shape.rank == 1:
            # 每个输出点都有不同的范围
            radii = 0.5 * extents
            self.nns = self.radius_search(
                inp_positions,
                queries=out_positions,
                radii=radii
            )
            if return_distances:
                neighbors_distance_normalized = self.nns.neighbors_distance_normalized
            else:
                neighbors_distance_normalized = None
        else:
            raise Exception("extents rank must be 0 or 1")

        # 提取邻居搜索结果
        # neighbors_index: [num_pairs] - 每个邻居对应的输入点索引
        # neighbors_row_splits: [num_output_points + 1] - 每个输出点的邻居范围
        neighbors_index = self.nns.neighbors_index
        neighbors_row_splits = self.nns.neighbors_row_splits

        # ==================== 步骤 2: 收集密度信息 ====================
        
        # 收集邻居点的密度
        # 形状: [num_pairs, 1]
        neighbor_densities = tf.gather(inp_densities, neighbors_index)
        
        # 计算每个邻居对应的查询点索引
        # num_neighbors_per_query: [num_output_points] - 每个查询点的邻居数量
        num_neighbors_per_query = neighbors_row_splits[1:] - neighbors_row_splits[:-1]
        
        # query_indices: [num_output_points] - 查询点索引序列 [0, 1, 2, ...]
        query_indices = tf.range(
            tf.shape(out_positions)[0],
            dtype=neighbors_row_splits.dtype
        )
        
        # query_point_idx_for_neighbors: [num_pairs] - 每个邻居对应的查询点索引
        # 例如: 如果查询点0有3个邻居，查询点1有2个邻居，则为 [0,0,0,1,1,...]
        query_point_idx_for_neighbors = tf.repeat(query_indices, num_neighbors_per_query)
        
        # 收集查询点的密度
        # 形状: [num_pairs, 1]
        query_densities_for_neighbors = tf.gather(
            out_densities,
            query_point_idx_for_neighbors
        )

        # ==================== 步骤 3: 密度条件化处理 ====================
        
        if self.density_modulation_mode == 'pairwise_film':
            # ========== Pair-wise FiLM 模式 ==========
            
            # 计算密度比
            # 形状: [num_pairs, 1]
            epsilon = 1e-6
            density_ratio = neighbor_densities / (query_densities_for_neighbors + epsilon)
            
            # NOTE:
            # ContinuousConv 的输入特征是按 input point 存储，不能直接注入
            # [num_pairs, in_channels] 的 pair-wise 特征。原先使用 scatter 会被重复
            # 覆盖，导致结果依赖邻居顺序且显存开销巨大。
            # 这里先把 pair-wise 密度比聚合成每个输入点的统计量，再执行点级 FiLM。
            num_input_points = tf.shape(inp_features)[0]
            density_ratio_sum = tf.math.unsorted_segment_sum(
                density_ratio,
                neighbors_index,
                num_segments=num_input_points
            )
            density_ratio_count = tf.math.unsorted_segment_sum(
                tf.ones_like(density_ratio),
                neighbors_index,
                num_segments=num_input_points
            )

            safe_count = tf.maximum(density_ratio_count, 1.0)
            density_ratio_per_input = density_ratio_sum / safe_count
            density_ratio_per_input = tf.where(
                density_ratio_count > 0.0,
                density_ratio_per_input,
                tf.ones_like(density_ratio_per_input)
            )

            # 生成点级 FiLM 参数并调制输入特征
            gamma_i = self.film_gamma_net(density_ratio_per_input)
            beta_i = self.film_beta_net(density_ratio_per_input)
            inp_features_modulated = gamma_i * inp_features + beta_i
            
            # 计算空间窗口权重（不包含密度比，因为已经在特征中体现）
            # custom_neighbors_importance: [num_pairs]
            if self._density_window_function is not None and neighbors_distance_normalized is not None:
                custom_neighbors_importance = self._density_window_function(
                    neighbors_distance_normalized,
                    neighbor_densities,
                    query_densities_for_neighbors
                )
            elif self.window_function is not None and neighbors_distance_normalized is not None:
                custom_neighbors_importance = self.window_function(neighbors_distance_normalized)
            else:
                custom_neighbors_importance = tf.ones(
                    [tf.shape(neighbors_index)[0]],
                    dtype=tf.float32
                )
            
            # 确保 importance 是秩 1 张量
            if custom_neighbors_importance.shape.rank == 2 and custom_neighbors_importance.shape[-1] == 1:
                custom_neighbors_importance = tf.squeeze(custom_neighbors_importance, axis=-1)
            
            # 调用父类卷积，使用调制后的特征
            out_features = super().call(
                inp_features=inp_features_modulated,  # 🔥 使用调制后的特征
                inp_positions=inp_positions,
                out_positions=out_positions,
                extents=extents,
                inp_importance=inp_importance,
                fixed_radius_search_hash_table=fixed_radius_search_hash_table,
                user_neighbors_index=neighbors_index,
                user_neighbors_row_splits=neighbors_row_splits,
                user_neighbors_importance=custom_neighbors_importance,
            )
            
        else:
            # ========== 密度比权重模式（默认）==========
            
            # 计算组合的邻居重要性权重
            # w = spatial_weight x density_weight
            # custom_neighbors_importance: [num_pairs]
            if self._density_window_function is not None and neighbors_distance_normalized is not None:
                custom_neighbors_importance = self._density_window_function(
                    neighbors_distance_normalized,
                    neighbor_densities,
                    query_densities_for_neighbors
                )
            elif self.window_function is not None and neighbors_distance_normalized is not None:
                custom_neighbors_importance = self.window_function(neighbors_distance_normalized)
            else:
                custom_neighbors_importance = tf.ones_like(neighbor_densities)
            
            # 确保 importance 是秩 1 张量
            if custom_neighbors_importance.shape.rank == 2 and custom_neighbors_importance.shape[-1] == 1:
                custom_neighbors_importance = tf.squeeze(custom_neighbors_importance, axis=-1)
            
            # 调用父类卷积，使用原始特征
            out_features = super().call(
                inp_features=inp_features,  # 使用原始特征
                inp_positions=inp_positions,
                out_positions=out_positions,
                extents=extents,
                inp_importance=inp_importance,
                fixed_radius_search_hash_table=fixed_radius_search_hash_table,
                user_neighbors_index=neighbors_index,
                user_neighbors_row_splits=neighbors_row_splits,
                user_neighbors_importance=custom_neighbors_importance,
            )

        return out_features


# ==================== 窗口函数定义 ====================

def spatial_window_fn(r_sqr: tf.Tensor) -> tf.Tensor:
    """
    标准 Poly6 空间窗口函数
    
    Args:
        r_sqr: 归一化距离平方 [num_pairs]
    
    Returns:
        空间权重 [num_pairs]
    """
    return tf.clip_by_value((1 - r_sqr)**3, 0, 1)


def relative_density_importance(normalized_distance: tf.Tensor,
                                neighbor_relative_densities: tf.Tensor,
                                query_relative_densities: tf.Tensor) -> tf.Tensor:
    """
    结合空间距离和相对密度的重要性函数
    
    用于 density_ratio 模式，计算公式: 
    w = (1 - r²)³ x (ρⱼ/ρᵢ)
    
    Args:
        normalized_distance: 归一化距离 (r²/R² for L2)
            形状: [num_pairs]
        neighbor_relative_densities: 邻居点相对密度 (ρⱼ/ρ_rest)
            形状: [num_pairs, 1]
        query_relative_densities: 查询点相对密度 (ρᵢ/ρ_rest)
            形状: [num_pairs, 1]
    
    Returns:
        组合重要性权重
            形状: [num_pairs]
    """
    # 空间衰减部分: (1 - r²)³
    spatial_weight = tf.clip_by_value((1 - normalized_distance)**3, 0, 1)
    
    # 密度权重部分: ρⱼ/ρᵢ
    epsilon = 1e-6
    density_ratio_weight = tf.squeeze(neighbor_relative_densities, axis=-1) / \
                          (tf.squeeze(query_relative_densities, axis=-1) + epsilon)
    
    # 组合权重
    combined_weight = spatial_weight * density_ratio_weight
    
    return combined_weight


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("多密度连续卷积层测试")
    print("=" * 60)
    
    # 创建测试数据
    num_input_points = 100
    num_output_points = 50
    in_channels = 4
    
    print(f"\n创建测试数据:")
    print(f"  - 输入点数: {num_input_points}")
    print(f"  - 输出点数: {num_output_points}")
    print(f"  - 输入通道数: {in_channels}")
    
    inp_features = tf.random.normal((num_input_points, in_channels))
    inp_positions = tf.random.normal((num_input_points, 3))
    out_positions = tf.random.normal((num_output_points, 3))
    extents = tf.constant(2.0, dtype=tf.float32)
    inp_densities = tf.random.uniform((num_input_points, 1), minval=0.5, maxval=10.0)
    out_densities = tf.random.uniform((num_output_points, 1), minval=0.5, maxval=10.0)
    
    print(f"\n张量形状:")
    print(f"  - inp_features: {inp_features.shape}")
    print(f"  - inp_positions: {inp_positions.shape}")
    print(f"  - inp_densities: {inp_densities.shape}")
    
    # ========== 测试 1: 密度比模式 ==========
    print("\n" + "=" * 60)
    print("测试 1: density_ratio 模式")
    print("=" * 60)
    
    conv_ratio = MultiDensityContinuousConv(
        filters=16,
        kernel_size=[4, 4, 4],
        coordinate_mapping='ball_to_cube_radial',
        interpolation='linear',
        normalize=True,
        window_function=None,
        combined_importance_function=relative_density_importance,
        density_modulation_mode='density_ratio'
    )
    
    conv_ratio.build(inp_features.shape)
    out_ratio = conv_ratio(
        inp_features, inp_positions, out_positions, extents,
        inp_densities, out_densities
    )
    
    print(f"✓ 输出形状: {out_ratio.shape}")
    print(f"✓ 邻居数统计:")
    num_neighbors = ml3d.ops.reduce_subarrays_sum(
        tf.ones_like(conv_ratio.nns.neighbors_index, dtype=tf.float32),
        conv_ratio.nns.neighbors_row_splits
    )
    print(f"  - 平均邻居数: {tf.reduce_mean(num_neighbors).numpy():.2f}")
    print(f"  - 最大邻居数: {tf.reduce_max(num_neighbors).numpy():.0f}")
    print(f"  - 最小邻居数: {tf.reduce_min(num_neighbors).numpy():.0f}")
    
    # ========== 测试 2: Pair-wise FiLM 模式 ==========
    print("\n" + "=" * 60)
    print("测试 2: pairwise_film 模式")
    print("=" * 60)
    
    conv_film = MultiDensityContinuousConv(
        filters=16,
        kernel_size=[4, 4, 4],
        coordinate_mapping='ball_to_cube_radial',
        interpolation='linear',
        normalize=True,
        window_function=None,
        combined_importance_function=relative_density_importance,
        density_modulation_mode='pairwise_film',
        film_hidden_dim=16
    )
    
    conv_film.build(inp_features.shape)
    out_film = conv_film(
        inp_features, inp_positions, out_positions, extents,
        inp_densities, out_densities
    )
    
    print(f"✓ 输出形状: {out_film.shape}")
    print(f"✓ FiLM 网络参数:")
    total_params = sum([tf.size(w).numpy() for w in conv_film.film_gamma_net.trainable_weights])
    total_params += sum([tf.size(w).numpy() for w in conv_film.film_beta_net.trainable_weights])
    print(f"  - 总参数数量: {total_params}")
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)
