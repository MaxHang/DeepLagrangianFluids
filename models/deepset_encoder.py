import tensorflow as tf

class DeepSetPhaseEncoder(tf.keras.Model):
    """
    使用 Deep Sets 架构将一个粒子上可变数量的相信息编码为固定大小的特征向量。
    """
    def __init__(self, phi_dims, rho_dims, aggregation='mean', name="DeepSetPhaseEncoder"):
        """
        Args:
            phi_dims (list of int): φ 网络的每个隐藏层和输出层的维度。
            rho_dims (list of int): ρ 网络的每个隐藏层和输出层的维度。
        """
        super().__init__(name=name)
        
        # φ 网络：处理单个相的特征
        self.phi_net = tf.keras.Sequential(name="phi_network")
        for dim in phi_dims:
            self.phi_net.add(tf.keras.layers.Dense(dim, activation='relu'))

        # 聚合特征的方式
        self.aggregation = aggregation
        
        # ρ 网络：处理聚合后的特征
        self.rho_net = tf.keras.Sequential(name="rho_network")
        for dim in rho_dims:
            self.rho_net.add(tf.keras.layers.Dense(dim, activation='relu'))
            
    def call(self, inputs):
        """
        Args:
            inputs (tuple): 包含两个张量
                - phase_features (tf.Tensor): 形状为 [N, max_num_phases, num_phase_features]
                  N 通常是 batch_size * num_particles
                - mask (tf.Tensor): 形状为 [N, max_num_phases]，布尔或浮点型掩码，
                  标记哪些是真实的相（1.0）哪些是填充（0.0）。
        
        Returns:
            tf.Tensor: 形状为 [N, rho_output_dim]，代表每个粒子的固定维度嵌入。
        """
        phase_features, mask = inputs
        
        # 1. 应用 φ 网络到每个相
        #    输入: [N, max_num_phases, num_phase_features]
        #    输出: [N, max_num_phases, phi_output_dim]
        # TimeDistributed 会将 phi_net 应用到时间步（这里是 max_num_phases 维度）上
        phi_output = tf.keras.layers.TimeDistributed(self.phi_net)(phase_features)
        
        # 2. 应用掩码，将填充部分的输出置零，以免影响求和
        #    我们需要将掩码扩展一个维度以进行广播
        mask_expanded = mask[..., tf.newaxis] # 形状变为 [N, max_num_phases, 1]
        masked_phi_output = phi_output * tf.cast(mask_expanded, dtype=phi_output.dtype)
        
        # 3. 聚合（置换不变操作）
        #    沿 'max_num_phases' 维度求和
        #    输出: [N, phi_output_dim]
        if self.aggregation == 'sum':
            aggregated_features = tf.reduce_sum(masked_phi_output, axis=1)
        elif self.aggregation == 'mean':
            # 3.1. 计算有效元素的数量
            num_valid_phases = tf.reduce_sum(tf.cast(mask, dtype=phi_output.dtype), axis=1, keepdims=True)
            # 3.2. 避免除以零
            num_valid_phases = tf.maximum(num_valid_phases, 1.0)
            # 3.3. 先求和，再除以数量
            sum_features = tf.reduce_sum(masked_phi_output, axis=1)
            aggregated_features = sum_features / num_valid_phases

        # # 新的做法：只对有效的元素求平均
        # # 3.1. 计算有效元素的数量
        # num_valid_phases = tf.reduce_sum(tf.cast(mask, dtype=phi_output.dtype), axis=1, keepdims=True)
        # # 3.2. 避免除以零
        # num_valid_phases = tf.maximum(num_valid_phases, 1.0)
        # # 3.3. 先求和，再除以数量
        # sum_features = tf.reduce_sum(masked_phi_output, axis=1)
        # aggregated_features = sum_features / num_valid_phases
        
        # 4. 应用 ρ 网络
        #    输入: [N, phi_output_dim]
        #    输出: [N, rho_output_dim]
        final_embedding = self.rho_net(aggregated_features)
        
        return final_embedding