import tensorflow as tf

class DeepSetPhaseEncoder(tf.keras.Model):
    """
    使用 Deep Sets 架构 + 显式循环处理可变数量的相。
    """
    def __init__(self, phi_dims: list, rho_dims: list, aggregation: str ='mean', name: str ="DeepSetPhaseEncoder"):
        super().__init__(name=name)
        
        # φ 网络：处理单个相的特征
        self.phi_net = tf.keras.Sequential(name="phi_network")
        for dim in phi_dims:
            self.phi_net.add(tf.keras.layers.Dense(dim, activation='relu'))

        self.aggregation = aggregation
        
        # ρ 网络：处理聚合后的特征
        self.rho_net = tf.keras.Sequential(name="rho_network")
        for dim in rho_dims:
            self.rho_net.add(tf.keras.layers.Dense(dim, activation='relu'))
            
    def call(self, inputs: tuple):
        """
        Args:
            inputs (tuple): 
                - phase_features (tf.Tensor): [N, num_phases, num_phase_features]
                  注意：这里num_phases是实际的相数，不需要是固定的max值
                - mask (tf.Tensor): [N, num_phases]，可选的掩码
        
        Returns:
            tf.Tensor: [N, rho_output_dim]
        """
        phase_features, mask = inputs
        
        # 获取实际的相数量（动态）
        num_phases = tf.shape(phase_features)[1]
        num_particles = tf.shape(phase_features)[0]
        
        # 方法1: 使用tf.map_fn (更Pythonic)
        # 对每个相应用phi网络
        def apply_phi_to_phase(phase_idx):
            # phase_features[:, phase_idx, :] 的形状是 [N, num_phase_features]
            phase_data = phase_features[:, phase_idx, :]
            return self.phi_net(phase_data)  # 输出 [N, phi_output_dim]
        
        # tf.map_fn会迭代num_phases次
        phi_outputs = tf.map_fn(
            apply_phi_to_phase,
            tf.range(num_phases),
            fn_output_signature=tf.TensorSpec(shape=[None, self.phi_net.layers[-1].units], dtype=tf.float32)
        )
        # phi_outputs 形状: [num_phases, N, phi_output_dim]
        
        # 转置回 [N, num_phases, phi_output_dim]
        phi_outputs = tf.transpose(phi_outputs, [1, 0, 2])
        
        # 应用掩码
        mask_expanded = mask[..., tf.newaxis]
        masked_phi_output = phi_outputs * tf.cast(mask_expanded, dtype=phi_outputs.dtype)
        
        # 聚合
        if self.aggregation == 'sum':
            aggregated_features = tf.reduce_sum(masked_phi_output, axis=1)
        elif self.aggregation == 'mean':
            num_valid_phases = tf.reduce_sum(tf.cast(mask, dtype=phi_outputs.dtype), axis=1, keepdims=True)
            num_valid_phases = tf.maximum(num_valid_phases, 1.0)
            sum_features = tf.reduce_sum(masked_phi_output, axis=1)
            aggregated_features = sum_features / num_valid_phases
        
        # 应用ρ网络
        final_embedding = self.rho_net(aggregated_features)
        
        return final_embedding