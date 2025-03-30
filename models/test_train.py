#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# 导入自定义网络
from models.default_tf_multi_mix_fluid import MultiPhaseParticleNetwork

def visualize_particles(pos, phase_fractions, title, save_path=None):
    """可视化粒子位置和相分布"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 根据体积分数确定颜色（使用RG颜色空间, R代表相1，G代表相2）
    if phase_fractions is not None:
        # 相1比例作为红色分量
        phase1 = phase_fractions[:, 0]
        # 相2比例计算为1-相1，作为绿色分量
        phase2 = 1.0 - phase1
        colors = np.column_stack([phase1, phase2, np.zeros_like(phase1)])
    else:
        # 如果没有相信息，全部标为蓝色
        colors = np.array([[0, 0, 1] for _ in range(len(pos))])
    
    # 绘制散点图
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=colors, s=30, alpha=0.8)
    
    # 设置图形属性
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    
    # 添加相分布图例
    if phase_fractions is not None:
        red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='相 1')
        green_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='相 2')
        ax.legend(handles=[red_patch, green_patch], loc='upper right')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def create_dam_break_scene(num_particles=1000, num_phases=2, mix_layers=True):
    """创建溃坝场景的初始条件
    
    Args:
        num_particles: 粒子总数
        num_phases: 相数量
        mix_layers: 是否在接触面附近创建混合层
    
    Returns:
        pos: 初始粒子位置
        vel: 初始粒子速度
        phase_fractions: 初始体积分数
        box: 边界粒子位置
        box_normals: 边界法线
    """
    # 创建容器边界（简化为立方体）
    box_size = 2.0
    box_points = []
    box_normals = []
    
    # 底部和侧面的边界点
    density = 0.1  # 边界点密度
    for x in np.arange(-box_size/2, box_size/2, density):
        for z in np.arange(-box_size/2, box_size/2, density):
            # 底部平面
            box_points.append([x, -box_size/2, z])
            box_normals.append([0, 1, 0])
            
    for y in np.arange(-box_size/2, box_size/2, density):
        for z in np.arange(-box_size/2, box_size/2, density):
            # 左右侧面
            box_points.append([-box_size/2, y, z])
            box_normals.append([1, 0, 0])
            box_points.append([box_size/2, y, z])
            box_normals.append([-1, 0, 0])
            
    for x in np.arange(-box_size/2, box_size/2, density):
        for y in np.arange(-box_size/2, box_size/2, density):
            # 前后侧面
            box_points.append([x, y, -box_size/2])
            box_normals.append([0, 0, 1])
            box_points.append([x, y, box_size/2])
            box_normals.append([0, 0, -1])
    
    # 创建流体粒子
    pos = []
    phase_fractions = [] if num_phases > 1 else None
    
    # 生成网格分布的流体粒子
    particles_per_dim = int(np.cbrt(num_particles))
    spacing = 0.03  # 粒子间距
    
    for i in range(particles_per_dim):
        for j in range(particles_per_dim):
            for k in range(particles_per_dim):
                # 在左半部分放置流体
                x = -box_size/2 + 0.1 + i * spacing
                y = -box_size/2 + 0.1 + j * spacing
                z = -box_size/2 + 0.1 + k * spacing
                
                if x < 0:  # 只在左半部分填充
                    pos.append([x, y, z])
                    
                    if num_phases > 1:
                        # 创建两层流体：底部是相1，顶部是相2
                        if y < -0.2:  # 底层是相1
                            if mix_layers and y > -0.3:  # 接触面附近创建混合层
                                # 线性混合
                                ratio = (y + 0.3) / 0.1  # 0到1之间的平滑过渡
                                phase_fractions.append([1.0 - ratio])
                            else:
                                phase_fractions.append([1.0])  # 纯相1
                        else:  # 顶层是相2
                            if mix_layers and y < -0.1:  # 接触面附近创建混合层
                                # 线性混合
                                ratio = (y + 0.2) / 0.1  # 0到1之间的平滑过渡
                                phase_fractions.append([0.0 + ratio])
                            else:
                                phase_fractions.append([0.0])  # 纯相2
    
    # 转换为numpy数组
    pos = np.array(pos, dtype=np.float32)
    if phase_fractions:
        phase_fractions = np.array(phase_fractions, dtype=np.float32)
    vel = np.zeros_like(pos, dtype=np.float32)  # 初始速度为零
    box = np.array(box_points, dtype=np.float32)
    box_normals = np.array(box_normals, dtype=np.float32)
    
    return pos, vel, phase_fractions, box, box_normals

def simulate_multi_phase_fluid(num_steps=100, visualize=True, save_results=False):
    """运行多相流体模拟
    
    Args:
        num_steps: 模拟步数
        visualize: 是否可视化结果
        save_results: 是否保存结果
    """
    # 创建输出目录
    if save_results:
        output_dir = "multi_phase_simulation_results"
        os.makedirs(output_dir, exist_ok=True)
    
    # 设置模拟参数
    num_phases = 2  # 相数量
    
    # 创建溃坝场景
    pos, vel, phase_fractions, box, box_normals = create_dam_break_scene(
        num_particles=1000, num_phases=num_phases, mix_layers=True)
    
    # print(f"创建了 {len(pos)} 个流体粒子和 {len(box)} 个边界粒子")
    print(f'created {len(pos)} fluid particles and {len(box)} boundary particles')
    
    # 初始化模型
    model = MultiPhaseParticleNetwork(
        particle_radius=0.025,
        num_phases=num_phases,
        timestep=1/30,  # 更新频率
        gravity=(0, -9.81, 0),  # 重力方向
        cd_cf_as_input=True
    )
    
    # 初始化网络权重

    model.init()
    model.summary()
    
    # 输入转换为张量
    tf_pos = tf.convert_to_tensor(pos, dtype=tf.float32)
    tf_vel = tf.convert_to_tensor(vel, dtype=tf.float32)
    tf_box = tf.convert_to_tensor(box, dtype=tf.float32)
    tf_box_normals = tf.convert_to_tensor(box_normals, dtype=tf.float32)
    
    if phase_fractions is not None:
        tf_phase_fractions = tf.convert_to_tensor(phase_fractions, dtype=tf.float32)
    else:
        tf_phase_fractions = None
    
    # 可视化初始状态
    if visualize:
        visualize_particles(pos, phase_fractions, 
                           f"initial status - {len(pos)} particles", 
                           save_path=os.path.join(output_dir, "step_0.png") if save_results else None)
    
    # 运行模拟
    start_time = time.time()
    
    # 设置交换系数和扩散系数
    cd = 0.7  # 交换系数
    cf = 0.3  # 扩散系数
    
    # 创建追踪每个相体积的列表
    phase_volumes = []
    
    for step in range(1, num_steps + 1):
        # 执行一步模拟
        inputs = (tf_pos, tf_vel, tf_phase_fractions, tf_box, tf_box_normals)
        print(f"step {step}/{num_steps} - cal time: {time.time() - start_time:.2f} s")
        print("Input pos shape: ", tf_pos.shape)
        print("Input vel shape: ", tf_vel.shape)
        print("Input phase_fractions shape: ", tf_phase_fractions.shape)
        
        if num_phases > 1:
            tf_pos, tf_vel, tf_phase_fractions = model(inputs, cd=cd, cf=cf)
        else:
            tf_pos, tf_vel = model(inputs)
        
        # 转换回numpy数组以便可视化
        pos = tf_pos.numpy()
        vel = tf_vel.numpy()
        
        if tf_phase_fractions is not None:
            phase_fractions = tf_phase_fractions.numpy()
            # 计算并记录每个相的总体积
            if len(phase_volumes) == 0:
                # 初始化跟踪数组
                phase_volumes = [[0] for _ in range(num_phases)]
            
            # 记录相1的总体积
            phase1_volume = np.sum(phase_fractions[:, 0])
            phase_volumes[0].append(phase1_volume)
            
            # 记录相2的总体积（1-相1）
            phase2_volume = len(phase_fractions) - phase1_volume
            phase_volumes[1].append(phase2_volume)
        
        # 每10步显示一次
        if visualize and step % 10 == 0:
            print(f"step {step}/{num_steps} - cal time: {time.time() - start_time:.2f} 秒")
            visualize_particles(pos, phase_fractions, 
                               f"step {step} - CD: {cd:.1f}, CF: {cf:.1f}", 
                               save_path=os.path.join(output_dir, f"step_{step}.png") if save_results else None)
            start_time = time.time()  # 重置定时器
    
    # 可视化相体积变化
    # if num_phases > 1 and len(phase_volumes[0]) > 0:
    #     plt.figure(figsize=(10, 6))
    #     steps = range(len(phase_volumes[0]))
        
    #     for i in range(num_phases):
    #         plt.plot(steps, phase_volumes[i], label=f"相 {i+1}")
            
    #     plt.xlabel('模拟步数')
    #     plt.ylabel('总体积')
    #     plt.title('各相总体积随时间变化')
    #     plt.legend()
    #     plt.grid(True)
        
    #     if save_results:
    #         plt.savefig(os.path.join(output_dir, "phase_volumes.png"))
    #     else:
    #         plt.show()
    
    print("sim done!")
    
    return model, pos, vel, phase_fractions

if __name__ == "__main__":
    # 运行多相流体模拟示例
    simulate_multi_phase_fluid(num_steps=50, visualize=False, save_results=False)