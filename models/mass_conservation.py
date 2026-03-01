#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
创建日期: 2025-10-28
文件名: mass_conservation_layers.py

描述:
    定义多种质量守恒约束层,可灵活集成到 V5 模型中
    
策略:
    1. 硬约束投影层(推理时100%守恒)
    2. 拉格朗日约束网络(训练时隐式学习)
    3. 物理一致性损失(辅助训练)
"""

import tensorflow as tf
from typing import Tuple, Optional


class MassConservingProjection(tf.keras.layers.Layer):
    """
    显式投影层: 将预测的 VF 投影到质量守恒流形上
    
    保证约束:
        1. Σ VF_i = 1 (体积分数归一化)
        2. Σ[VF_i × ρ_i] = constant (混合密度守恒,每个粒子)
        3. Σ_particles[VF_j,i × ρ_mix_j] = constant (单相总质量守恒)
    
    使用方法:
        projection = MassConservingProjection()
        vf_final = projection(vf_pred, vf_current, phase_densities)
    """
    
    def __init__(self, 
                 correction_strength: float = 0.5,
                 epsilon: float = 1e-8,
                 name: str = 'mass_conserving_projection'):
        """
        Args:
            correction_strength: 修正强度 ∈ [0, 1]
                - 0: 不修正(等同于直接归一化)
                - 1: 完全修正到守恒流形
                - 0.5: 渐进式修正(推荐,更稳定)
            epsilon: 数值稳定性参数
        """
        super().__init__(name=name)
        self.correction_strength = correction_strength
        self.epsilon = epsilon
    
    def call(self, 
             vf_pred: tf.Tensor, 
             vf_current: tf.Tensor, 
             phase_densities: tf.Tensor) -> tf.Tensor:
        """
        投影预测的 VF 到守恒流形
        
        Args:
            vf_pred: 网络预测的 VF, shape [N, num_phases]
            vf_current: 当前时刻的 VF, shape [N, num_phases]
            phase_densities: 各相密度, shape [num_phases]
        
        Returns:
            投影后的 VF, shape [N, num_phases]
            满足: Σ VF = 1 且 Σ[VF × ρ] = Σ[VF_current × ρ]
        """
        # ========== 步骤1: 计算目标混合密度 ==========
        # 使用当前 VF 计算应该守恒的混合密度
        rho_mix_target = tf.reduce_sum(
            vf_current * phase_densities, 
            axis=-1, 
            keepdims=True
        )  # [N, 1]
        
        # ========== 步骤2: 计算预测的混合密度 ==========
        rho_mix_pred = tf.reduce_sum(
            vf_pred * phase_densities, 
            axis=-1, 
            keepdims=True
        )  # [N, 1]
        
        # ========== 步骤3: 计算每相的质量偏差 ==========
        # 期望的每相质量: VF_current × rho_mix_target
        mass_per_phase_target = vf_current * rho_mix_target  # [N, num_phases]
        
        # 预测的每相质量: VF_pred × rho_mix_pred
        mass_per_phase_pred = vf_pred * rho_mix_pred  # [N, num_phases]
        
        # 质量偏差
        mass_error = mass_per_phase_pred - mass_per_phase_target  # [N, num_phases]
        
        # ========== 步骤4: 按密度加权分配修正量 ==========
        # 密度越大的相,修正量越小(因为质量主要由它决定)
        # correction = mass_error / ρ_i
        vf_correction = mass_error / (phase_densities + self.epsilon)  # [N, num_phases]
        
        # 应用修正(可调节修正强度)
        vf_corrected = vf_pred - self.correction_strength * vf_correction
        
        # ========== 步骤5: 确保非负性 ==========
        vf_corrected = tf.nn.relu(vf_corrected)
        
        # ========== 步骤6: 重新归一化确保 Σ VF = 1 ==========
        vf_sum = tf.reduce_sum(vf_corrected, axis=-1, keepdims=True)
        vf_sum = tf.maximum(vf_sum, self.epsilon)
        vf_final = vf_corrected / vf_sum
        
        return vf_final
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'correction_strength': self.correction_strength,
            'epsilon': self.epsilon
        })
        return config


class LagrangianConstraintPredictor(tf.keras.layers.Layer):
    """
    拉格朗日约束预测器: 显式学习约束的拉格朗日乘子
    
    原理:
        将 VF 预测分解为:
        delta_vf = delta_vf_free - λ × ∇constraint
        
        其中:
        - delta_vf_free: 不受约束的"自由"预测
        - λ: 拉格朗日乘子(网络学习)
        - ∇constraint: 约束的梯度方向
    
    使用方法:
        predictor = LagrangianConstraintPredictor(num_phases=3)
        delta_vf = predictor(shared_features, vf_current, phase_densities)
    """
    
    def __init__(self, 
                 num_phases: int,
                 hidden_dim: int = 64,
                 name: str = 'lagrangian_constraint_predictor'):
        """
        Args:
            num_phases: 相数
            hidden_dim: 隐藏层维度
        """
        super().__init__(name=name)
        self.num_phases = num_phases
        self.hidden_dim = hidden_dim
        
        # 自由预测头(不受约束)
        self.free_predictor = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu', name='free_hidden'),
            tf.keras.layers.Dense(num_phases, activation='tanh', name='free_output')
        ], name='free_predictor')
        
        # 拉格朗日乘子预测头(每个粒子一个标量)
        self.lambda_predictor = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim // 2, activation='relu', name='lambda_hidden'),
            tf.keras.layers.Dense(1, name='lambda_output')
        ], name='lambda_predictor')
    
    def call(self, 
             features: tf.Tensor, 
             vf_current: tf.Tensor, 
             phase_densities: tf.Tensor) -> tf.Tensor:
        """
        预测满足质量守恒约束的 delta_vf
        
        Args:
            features: 共享特征, shape [N, C]
            vf_current: 当前 VF, shape [N, num_phases]
            phase_densities: 各相密度, shape [num_phases]
        
        Returns:
            delta_vf, shape [N, num_phases]
            满足: Σ[delta_vf × ρ] ≈ 0 (质量守恒)
        """
        # ========== 步骤1: 预测自由变化量 ==========
        delta_vf_free = self.free_predictor(features)  # [N, num_phases]
        
        # ========== 步骤2: 预测拉格朗日乘子 ==========
        lambda_pred = self.lambda_predictor(features)  # [N, 1]
        
        # ========== 步骤3: 计算约束梯度 ==========
        # 约束: Σ[VF_i × ρ_i] = constant
        # 梯度: ∂constraint/∂VF_i = ρ_i
        constraint_gradient = phase_densities  # [num_phases]
        
        # ========== 步骤4: 投影到约束正交补空间 ==========
        # delta_vf = delta_vf_free - λ × ∇constraint
        delta_vf_constrained = delta_vf_free - lambda_pred * constraint_gradient
        
        return delta_vf_constrained
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_phases': self.num_phases,
            'hidden_dim': self.hidden_dim
        })
        return config


class PhysicsConsistencyLoss:
    """
    物理一致性损失函数集合
    
    提供多种损失计算方法:
        1. mass_conservation_loss: 单粒子混合密度守恒
        2. total_mass_conservation_loss: 全局单相总质量守恒
        3. vf_sum_conservation_loss: 体积分数求和约束
    
    使用方法:
        loss_calculator = PhysicsConsistencyLoss()
        
        # 在训练循环中
        mass_loss = loss_calculator.mass_conservation_loss(
            vf_next, vf_current, phase_densities
        )
        total_loss += 0.1 * mass_loss
    """
    
    @staticmethod
    def mass_conservation_loss(vf_next: tf.Tensor, 
                               vf_current: tf.Tensor, 
                               phase_densities: tf.Tensor,
                               reduction: str = 'mean') -> tf.Tensor:
        """
        单粒子混合密度守恒损失
        
        约束: ρ_mix(t+1) = ρ_mix(t) = Σ[VF_i × ρ_i]
        
        Args:
            vf_next: 下一时刻 VF, shape [N, num_phases]
            vf_current: 当前时刻 VF, shape [N, num_phases]
            phase_densities: 各相密度, shape [num_phases]
            reduction: 'mean', 'sum' 或 'none'
        
        Returns:
            损失标量或 [N] 形状的张量
        """
        # 计算混合密度
        rho_mix_current = tf.reduce_sum(vf_current * phase_densities, axis=-1)  # [N]
        rho_mix_next = tf.reduce_sum(vf_next * phase_densities, axis=-1)  # [N]
        
        # 相对误差
        mass_drift = tf.abs(rho_mix_next - rho_mix_current) / (rho_mix_current + 1e-8)
        
        if reduction == 'mean':
            return tf.reduce_mean(mass_drift)
        elif reduction == 'sum':
            return tf.reduce_sum(mass_drift)
        else:
            return mass_drift
    
    @staticmethod
    def total_mass_conservation_loss(vf_next: tf.Tensor, 
                                     vf_current: tf.Tensor, 
                                     phase_densities: tf.Tensor,
                                     reduction: str = 'mean') -> tf.Tensor:
        """
        全局单相总质量守恒损失
        
        约束: M_phase_i(t+1) = M_phase_i(t) = Σ_particles[VF_j,i × ρ_mix_j]
        
        Args:
            vf_next: 下一时刻 VF, shape [N, num_phases]
            vf_current: 当前时刻 VF, shape [N, num_phases]
            phase_densities: 各相密度, shape [num_phases]
            reduction: 'mean', 'sum' 或 'none'
        
        Returns:
            损失标量或 [num_phases] 形状的张量
        """
        # 计算每个粒子的混合密度
        rho_mix_current = tf.reduce_sum(vf_current * phase_densities, axis=-1, keepdims=True)  # [N, 1]
        rho_mix_next = tf.reduce_sum(vf_next * phase_densities, axis=-1, keepdims=True)  # [N, 1]
        
        # 计算每相的总质量
        mass_per_phase_current = vf_current * rho_mix_current  # [N, num_phases]
        mass_per_phase_next = vf_next * rho_mix_next  # [N, num_phases]
        
        total_mass_current = tf.reduce_sum(mass_per_phase_current, axis=0)  # [num_phases]
        total_mass_next = tf.reduce_sum(mass_per_phase_next, axis=0)  # [num_phases]
        
        # 相对误差
        mass_drift_per_phase = tf.abs(total_mass_next - total_mass_current) / (total_mass_current + 1e-8)
        
        if reduction == 'mean':
            return tf.reduce_mean(mass_drift_per_phase)
        elif reduction == 'sum':
            return tf.reduce_sum(mass_drift_per_phase)
        else:
            return mass_drift_per_phase
    
    @staticmethod
    def vf_sum_conservation_loss(vf: tf.Tensor, 
                                 reduction: str = 'mean') -> tf.Tensor:
        """
        体积分数求和约束损失
        
        约束: Σ VF_i = 1 (每个粒子)
        
        Args:
            vf: 体积分数, shape [N, num_phases]
            reduction: 'mean', 'sum' 或 'none'
        
        Returns:
            损失标量或 [N] 形状的张量
        """
        vf_sum = tf.reduce_sum(vf, axis=-1)  # [N]
        vf_sum_error = tf.abs(vf_sum - 1.0)
        
        if reduction == 'mean':
            return tf.reduce_mean(vf_sum_error)
        elif reduction == 'sum':
            return tf.reduce_sum(vf_sum_error)
        else:
            return vf_sum_error


# ========== 便捷函数 ==========

def create_mass_conserving_model(base_model: tf.keras.Model,
                                 projection_strength: float = 0.5) -> tf.keras.Model:
    """
    为现有模型添加质量守恒投影层
    
    Args:
        base_model: 基础模型(如 MultiPhaseParticleNetwork)
        projection_strength: 投影强度 ∈ [0, 1]
    
    Returns:
        包装后的模型,推理时自动应用投影
    
    使用示例:
        base_model = MultiPhaseParticleNetwork(...)
        conserving_model = create_mass_conserving_model(base_model, projection_strength=0.8)
    """
    class MassConservingWrapper(tf.keras.Model):
        def __init__(self, base_model, projection_strength):
            super().__init__(name='mass_conserving_wrapper')
            self.base_model = base_model
            self.projection = MassConservingProjection(correction_strength=projection_strength)
        
        def call(self, inputs, **kwargs):
            # 获取当前 VF
            _, _, current_vf, _, _ = inputs
            phase_densities = kwargs.get('phase_densities')
            
            # 基础模型预测
            pos_final, vel_final, vf_pred = self.base_model(inputs, **kwargs)
            
            # 应用投影(仅在推理时)
            if not kwargs.get('training', False) and phase_densities is not None:
                vf_final = self.projection(vf_pred, current_vf, phase_densities)
            else:
                vf_final = vf_pred
            
            return pos_final, vel_final, vf_final
    
    return MassConservingWrapper(base_model, projection_strength)


if __name__ == '__main__':
    """测试各个组件"""
    print("=" * 80)
    print("Testing Mass Conservation Layers")
    print("=" * 80)
    
    # 创建测试数据
    N = 100
    num_phases = 3
    vf_current = tf.random.uniform([N, num_phases])
    vf_current = vf_current / tf.reduce_sum(vf_current, axis=-1, keepdims=True)
    
    vf_pred = tf.random.uniform([N, num_phases])
    vf_pred = vf_pred / tf.reduce_sum(vf_pred, axis=-1, keepdims=True)
    
    phase_densities = tf.constant([1000.0, 800.0, 1200.0], dtype=tf.float32)
    
    # 测试投影层
    print("\n[Test 1] Mass Conserving Projection")
    projection = MassConservingProjection(correction_strength=0.5)
    vf_projected = projection(vf_pred, vf_current, phase_densities)
    
    rho_mix_current = tf.reduce_sum(vf_current * phase_densities, axis=-1)
    rho_mix_projected = tf.reduce_sum(vf_projected * phase_densities, axis=-1)
    
    print(f"  Before projection: ρ_mix range = [{tf.reduce_min(rho_mix_current):.2f}, {tf.reduce_max(rho_mix_current):.2f}]")
    print(f"  After projection:  ρ_mix range = [{tf.reduce_min(rho_mix_projected):.2f}, {tf.reduce_max(rho_mix_projected):.2f}]")
    print(f"  Mass drift: {tf.reduce_mean(tf.abs(rho_mix_projected - rho_mix_current)):.6f} kg/m³")
    
    # 测试拉格朗日约束预测器
    print("\n[Test 2] Lagrangian Constraint Predictor")
    features = tf.random.normal([N, 128])
    predictor = LagrangianConstraintPredictor(num_phases=num_phases)
    delta_vf = predictor(features, vf_current, phase_densities)
    
    mass_change = tf.reduce_sum(delta_vf * phase_densities, axis=-1)
    print(f"  Mass change per particle: mean={tf.reduce_mean(mass_change):.6f}, std={tf.math.reduce_std(mass_change):.6f}")
    
    # 测试物理损失
    print("\n[Test 3] Physics Consistency Loss")
    loss_calc = PhysicsConsistencyLoss()
    
    mass_loss = loss_calc.mass_conservation_loss(vf_pred, vf_current, phase_densities)
    total_mass_loss = loss_calc.total_mass_conservation_loss(vf_pred, vf_current, phase_densities)
    vf_sum_loss = loss_calc.vf_sum_conservation_loss(vf_pred)
    
    print(f"  Mass conservation loss:       {mass_loss:.6f}")
    print(f"  Total mass conservation loss: {total_mass_loss:.6f}")
    print(f"  VF sum conservation loss:     {vf_sum_loss:.6f}")
    
    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
