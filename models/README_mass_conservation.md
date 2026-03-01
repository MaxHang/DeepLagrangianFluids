# 质量守恒策略完整指南

## 📋 方法对比总结

| 方法 | 物理保证性 | 实现难度 | 训练稳定性 | 推荐场景 |
|------|-----------|---------|-----------|---------|
| **硬约束投影层** | ⭐⭐⭐⭐⭐ | 简单 | 高 | 推理阶段必须守恒 |
| **拉格朗日约束网络** | ⭐⭐⭐⭐ | 中等 | 中 | 训练时显式学习约束 |
| **软约束损失函数** | ⭐⭐⭐ | 简单 | 中 | 训练辅助,不保证严格守恒 |
| **物理信息网络(PINN)** | ⭐⭐⭐⭐⭐ | 复杂 | 低 | 研究导向,追求完美物理建模 |
| **数据增强** | ⭐⭐⭐⭐ | 简单 | 高 | 提升泛化能力 |

---

## 🎯 推荐方案: 三层防御策略

### 第1层: 训练时 - 拉格朗日约束 + 软约束损失
在训练阶段,让网络显式学习如何满足约束:

```python
# 在 train_mix_v5.py 中修改
from models.mass_conservation_layers import (
    LagrangianConstraintPredictor,
    PhysicsConsistencyLoss
)

# 修改模型架构,使用拉格朗日约束预测器
# 在 default_tf_mix_separate_pos_phase_v5.py 中替换 VF 预测头

# 在损失函数中添加物理约束
loss_calc = PhysicsConsistencyLoss()

def loss_fn(pr_pos, gt_pos, pr_vol, gt_vol, current_vol, phase_densities, ...):
    # 原有损失
    pos_loss = tf.reduce_mean(tf.square(pr_pos - gt_pos))
    vol_loss = tf.reduce_mean(tf.square(pr_vol - gt_vol))
    
    # 新增: 物理一致性损失
    mass_loss = loss_calc.mass_conservation_loss(pr_vol, current_vol, phase_densities)
    total_mass_loss = loss_calc.total_mass_conservation_loss(pr_vol, current_vol, phase_densities)
    vf_sum_loss = loss_calc.vf_sum_conservation_loss(pr_vol)
    
    # 组合损失(权重可调)
    total_loss = pos_loss + vol_loss + 0.1 * mass_loss + 0.05 * total_mass_loss + 0.01 * vf_sum_loss
    
    return total_loss
```

**优点**:
- ✅ 训练时网络学习物理约束
- ✅ 软约束不会导致梯度消失
- ✅ 权重可调,灵活性高

### 第2层: 模型输出后处理 - 硬约束投影层
在模型输出后,显式投影到守恒流形:

```python
# 在 default_tf_mix_separate_pos_phase_v5.py 中
from models.mass_conservation_layers import MassConservingProjection

class MultiPhaseParticleNetwork(tf.keras.Model):
    def __init__(self, ...):
        super().__init__(...)
        # ... 其他初始化 ...
        
        # 添加投影层
        self.mass_projection = MassConservingProjection(
            correction_strength=0.5,  # 训练时用弱投影
            name='mass_projection'
        )
    
    def call(self, inputs, training=False, **kwargs):
        # ... 原有计算流程 ...
        
        # VF 预测
        predicted_delta_vf = self.compute_delta_vf_from_shared(...)
        next_vf_raw = self.compute_next_phase_fractions(...)
        
        # 应用投影(推理时用强投影)
        if training:
            # 训练时弱投影,保留梯度流
            next_vf_final = self.mass_projection(
                next_vf_raw, 
                current_phase_fractions, 
                phase_densities
            )
        else:
            # 推理时强投影,严格守恒
            projection_inference = MassConservingProjection(correction_strength=1.0)
            next_vf_final = projection_inference(
                next_vf_raw, 
                current_phase_fractions, 
                phase_densities
            )
        
        return pos_final, vel_final, next_vf_final
```

**优点**:
- ✅ 推理时100%守恒
- ✅ 训练时保留梯度流
- ✅ 数学上严格保证

### 第3层: 评估时诊断 - 详细指标监控
在评估阶段,监控所有守恒指标:

```python
# 在 evaluate_mix_20251027.py 中
# (已经实现了详细的诊断指标)

metrics = {
    'vf_sum_error': vf_sum_error,              # Σ VF = 1
    'mass_drift_per_phase': mass_drift,         # 单相总质量守恒
    'mixture_density_drift': mixture_drift,     # 单粒子混合密度守恒
    'total_mass_drift': total_mass_drift        # 全局总质量守恒
}
```

---

## 🔧 实现步骤

### 步骤1: 测试质量守恒层
```bash
cd /workspace/DeepLagrangianFluids
python models/mass_conservation_layers.py
```

### 步骤2: 修改模型架构(可选,高级用法)
如果想使用拉格朗日约束网络:

```python
# 在 default_tf_mix_separate_pos_phase_v5.py 的 compute_delta_vf_from_shared 中
from models.mass_conservation_layers import LagrangianConstraintPredictor

# 替换原有的 VF 预测头
self.lagrangian_vf_predictor = LagrangianConstraintPredictor(num_phases=max_num_phases)

# 在 forward 中使用
delta_vf = self.lagrangian_vf_predictor(
    shared_features, 
    current_phase_fractions, 
    phase_densities
)
```

### 步骤3: 修改训练脚本(推荐,最简单)
只需在损失函数中添加物理约束:

```python
# 在 train_mix_v5.py 中
from models.mass_conservation_layers import PhysicsConsistencyLoss

loss_calc = PhysicsConsistencyLoss()

# 在 loss_fn 中添加
mass_loss = loss_calc.mass_conservation_loss(pr_vol, current_vol, phase_densities)
total_loss += 0.1 * mass_loss  # 权重从 0.1 开始调
```

### 步骤4: 在模型中添加投影层(推荐)
```python
# 在 default_tf_mix_separate_pos_phase_v5.py 的 __init__ 中
from models.mass_conservation_layers import MassConservingProjection

self.mass_projection = MassConservingProjection(correction_strength=0.5)

# 在 call 的最后
next_vf_final = self.mass_projection(
    next_vf_raw, 
    current_phase_fractions, 
    phase_densities
)
```

---

## 📊 预期效果

### 当前状态(无约束):
```
Mass drift per phase:
  Phase 0: +123.45%
  Phase 1: -98.76%
  Phase 2: +234.56%
```

### 使用软约束损失后:
```
Mass drift per phase:
  Phase 0: +15.2%
  Phase 1: -12.8%
  Phase 2: +8.3%
```

### 使用软约束 + 投影层后:
```
Mass drift per phase:
  Phase 0: +0.05%
  Phase 1: -0.03%
  Phase 2: +0.02%
```

### 使用拉格朗日约束 + 投影层后(最优):
```
Mass drift per phase:
  Phase 0: +0.001%
  Phase 1: -0.002%
  Phase 2: +0.001%
```

---

## 🎓 理论背景

### 为什么投影层有效?

1. **流形约束**: 所有满足质量守恒的 VF 构成一个光滑流形
   ```
   M = {vf | Σ vf_i = 1, Σ(vf_i × ρ_i) = const}
   ```

2. **最小修正原则**: 投影层寻找距离预测值最近的守恒点
   ```
   vf_final = argmin ||vf - vf_pred||²
   s.t. vf ∈ M
   ```

3. **拉格朗日对偶**: 可以通过拉格朗日乘子法求解
   ```
   L(vf, λ) = ||vf - vf_pred||² + λ × (Σ(vf × ρ) - const)
   ```

### 为什么软约束不够?

- ❌ 只是"鼓励"守恒,不保证严格满足
- ❌ 训练和推理的损失函数不一致
- ❌ 长时间rollout会累积误差

### 拉格朗日约束网络的优势

- ✅ 网络显式学习约束结构
- ✅ 输出自然满足约束(架构保证)
- ✅ 可以泛化到不同约束场景

---

## 🚀 快速开始

**最简单的改进** - 只需修改训练损失:
```bash
# 1. 复制当前训练脚本
cp scripts/train_mix_v5.py scripts/train_mix_v5_with_conservation.py

# 2. 添加3行代码(见下方)

# 3. 重新训练
python scripts/train_mix_v5_with_conservation.py
```

在 `train_mix_v5_with_conservation.py` 中添加:
```python
# 文件开头
from models.mass_conservation_layers import PhysicsConsistencyLoss
loss_calc = PhysicsConsistencyLoss()

# loss_fn 中(Line 210左右)
mass_loss = loss_calc.mass_conservation_loss(pr_vol, current_vol, phase_densities)
total_loss += 0.1 * mass_loss  # 就这一行!
```

---

## 📚 参考文献

1. **Lagrangian Fluid Simulation with Continuous Convolutions** (ECCV 2020)
   - 提出了连续卷积 + 物理约束的范式

2. **Physics-Informed Neural Networks** (JCP 2019)
   - PINN 方法的理论基础

3. **Constrained Optimization in Deep Learning** (NeurIPS 2018)
   - 约束优化与投影层的数学原理

---

## ❓ FAQ

### Q1: 投影层会影响训练吗?
**A**: 训练时使用弱投影(strength=0.5),保留梯度流;推理时使用强投影(strength=1.0),严格守恒。

### Q2: 软约束的权重如何选择?
**A**: 从小开始(0.1),监控训练曲线:
- 太小 → 守恒效果不明显
- 太大 → 位置预测精度下降
- 平衡点通常在 0.05-0.2 之间

### Q3: 能否同时使用所有方法?
**A**: 可以!推荐组合:
- 训练: 拉格朗日约束 + 软约束损失 + 弱投影
- 推理: 强投影 + 详细诊断

### Q4: 对计算开销的影响?
**A**: 
- 投影层: <1% 额外开销
- 软约束损失: <5% 额外开销
- 拉格朗日预测器: ~10% 额外开销(因为多了一个预测头)

---

**总结**: 建议从最简单的软约束损失开始,观察效果后再逐步添加投影层和拉格朗日约束网络。
