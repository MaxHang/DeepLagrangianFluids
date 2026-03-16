#!/usr/bin/env python3
import os
import sys
import argparse
import h5py
import numpy as np
import re
from glob import glob
import time
import importlib
import json
import tensorflow as tf
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
from physics_data_helper import numpy_from_bgeo, write_bgeo_from_numpy
from create_physics_scenes import obj_surface_to_particles, obj_volume_to_particles
import open3d as o3d
import plyfile
import yaml

# 在创建模型后添加
def print_model_structure(model):
    """Print model structure information in text format"""
    print("\n======= MODEL STRUCTURE SUMMARY =======")
    
    # Use built-in summary method
    model.summary()
    
    # Print network layer information and parameters
    print("\nLayer Details:")
    total_params = 0
    trainable_params = 0
    for layer_idx, layer in enumerate(model.layers):
        layer_name = getattr(layer, 'name', f'layer_{layer_idx}')
        print(f"Layer {layer_idx}: {layer_name}")
        
        # Print parameters for each layer
        layer_params = 0
        if hasattr(layer, 'trainable_variables'):
            for var in layer.trainable_variables:
                params = np.prod(var.shape)
                layer_params += params
                trainable_params += params
                print(f"  - {var.name}: {var.shape} = {params:,} params")
                
        print(f"  Total params in layer: {layer_params:,}")
        total_params += layer_params
    
    # Print network architecture specifics if available
    if hasattr(model, 'layer_channels'):
        print(f"\nChannel configuration: {model.layer_channels}")
    
    if hasattr(model, '_all_convs'):
        print("\nConvolution layers:")
        for name, _ in model._all_convs:
            print(f"  - {name}")
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("===================================\n")


def read_pos_vel_from_h5(path, frame_id=1, random_rotation=False):
    """Load h5py data files from specified path."""
    with h5py.File(path, 'r') as h5f:
        box = h5f['box'][:]
        box_normals = h5f['box_normals'][:]
        frame_group = h5f[f'frames/{frame_id}']  # 取第一帧
        pos = frame_group['pos'][:]
        vel = frame_group['vel'][:]
        density = frame_group['rest_density'][:]
    return [box, box_normals, pos, vel, density]

def write_particles(path_without_ext, pos, vel=None, density=None, options=None):
    """Writes the particles as point cloud ply.
    Optionally writes particles as bgeo which also supports velocities.
    """
    arrs = {'pos': pos}
    if vel is not None:
        arrs['vel'] = vel
    if density is not None:
        arrs['density'] = density
    np.savez(path_without_ext + '.npz', **arrs)

    if options and options.write_ply:
        # 准备需要写入到PLY的数据
        num_particles = pos.shape[0]
        
        # 准备plyfile所需的数据结构
        vertex_data = []
        
        # 添加位置数据 (x, y, z)
        vertex_data.append(('x', pos[:, 0].astype('float32')))
        vertex_data.append(('y', pos[:, 1].astype('float32')))
        vertex_data.append(('z', pos[:, 2].astype('float32')))
        
        # 如果有速度数据，添加速度 (vx, vy, vz)
        if vel is not None:
            vertex_data.append(('vx', vel[:, 0].astype('float32')))
            vertex_data.append(('vy', vel[:, 1].astype('float32')))
            vertex_data.append(('vz', vel[:, 2].astype('float32')))

        # 根据密度设置颜色
        if density is not None and density.shape[0] == num_particles:
            # 归一化密度到 [0, 1] 范围
            min_dens = np.min(density)
            max_dens = np.max(density)
            if max_dens > min_dens:
                norm_density = (density - min_dens) / (max_dens - min_dens)
            else:
                norm_density = np.ones(num_particles) * 0.5 # 如果密度都一样，设为中间值

            # 简单的颜色映射: 蓝色 (低密度) -> 红色 (高密度)
            colors = np.zeros((num_particles, 3), dtype=np.float32)
            colors[:, 0] = norm_density  # 红色通道
            colors[:, 2] = 1.0 - norm_density # 蓝色通道

            # 将颜色值转换为0-255范围的整数
            colors = (colors * 255).astype(np.uint8)
        else:
            # 默认颜色 (灰色)
            colors = np.full((num_particles, 3), 128, dtype=np.uint8)

        # 添加RGB颜色通道
        vertex_data.append(('red', colors[:, 0]))
        vertex_data.append(('green', colors[:, 1]))
        vertex_data.append(('blue', colors[:, 2]))

        # 创建vertex元素
        vertex_element = plyfile.PlyElement.describe(
            np.array(list(zip(*[data for _, data in vertex_data])),
                     dtype=[(name, data.dtype.str) for name, data in vertex_data]),
            'vertex'
        )

        # 创建PLY文件
        ply_data = plyfile.PlyData([vertex_element], text=True)

        # 写入PLY文件
        ply_data.write(path_without_ext + '.ply')

    if options and options.write_bgeo:
        # bgeo格式通常只支持位置和速度
        write_bgeo_from_numpy(path_without_ext + '.bgeo', pos, vel)


def run_sim_tf(trainscript_module, cfg, weights_path, scene, num_steps, output_dir,
               options, gpu='0'):

    # init the network
    model = trainscript_module.create_model(gpu, **cfg.get('model', {}))
    model.init()
    # 支持ckpt和h5两种权重格式
    if weights_path.endswith('.ckpt') or weights_path.endswith('.index'):
        checkpoint = tf.train.Checkpoint(model=model)
        restore_path = weights_path
        if restore_path.endswith('.index'):
            restore_path = restore_path[:-6]
        print(f"Restoring from checkpoint: {restore_path}")
        checkpoint.restore(restore_path).expect_partial()
    else:
        model.load_weights(weights_path, by_name=True)

    print_model_structure(model)

    fluids = []
    print(scene.keys())
    if 'h5_path' in scene:
        print(scene['h5_path'])
        frame_id = 1
        if 'frame_id' in scene:
            frame_id = scene['frame_id']
        data = read_pos_vel_from_h5(scene['h5_path'], frame_id, random_rotation=True)
        box, box_normals, points, velocities, density = data
        x = scene['fluids'][0]
        range_ = range(x['start'], x['stop'], x['step'])
        # feats/phase_fractions 这里为 None
        fluids.append((points, velocities, density, None, range_))
    else:
        # prepare static particles
        walls = []
        for x in scene['walls']:
            points, normals = obj_surface_to_particles(x['path'])
            if 'invert_normals' in x and x['invert_normals']:
                normals = -normals
            points += np.asarray([x['translation']], dtype=np.float32)
            walls.append((points, normals))
        box = np.concatenate([x[0] for x in walls], axis=0)
        box_normals = np.concatenate([x[1] for x in walls], axis=0)
        # prepare fluids
        for x in scene['fluids']:
            if 'h5_path' in x and os.path.exists(x['h5_path']):
                data = read_pos_vel_from_h5(x['h5_path'])
                points, velocities, density = data[2], data[3], data[4]
            else:
                points = obj_volume_to_particles(x['path'])[0]
                points += np.asarray([x['translation']], dtype=np.float32)
                velocities = np.empty_like(points)
                velocities[:, 0] = x['velocity'][0]
                velocities[:, 1] = x['velocity'][1]
                velocities[:, 2] = x['velocity'][2]
                # 如果配置中指定了相体积分数，使用它
                density = np.ones(points.shape[0], dtype=np.float32) * 1000.0
            
            # 获取扩散和交换系数
            feats = None
            range_ = range(x['start'], x['stop'], x['step'])
            fluids.append((points, velocities, density, feats, range_))
    
    # compute lowest point for removing out of bounds particles
    min_y = np.min(box[:, 1]) - 0.05 * (np.max(box[:, 1]) - np.min(box[:, 1]))
    # export static particles
    write_particles(os.path.join(output_dir, 'box'), box, box_normals, None, options)

    pos = np.empty(shape=(0, 3), dtype=np.float32)
    vel = np.empty_like(pos)
    density = np.empty(shape=(0,), dtype=np.float32)
    feats = None

    start_time = time.time()
    for step in range(num_steps):
        # add from fluids to pos vel arrays
        for points, velocities, fluid_density, feats, range_ in fluids:
            if step in range_:  # check if we have to add the fluid at this point in time
                pos = np.concatenate([pos, points], axis=0)
                vel = np.concatenate([vel, velocities], axis=0)
                density = np.concatenate([density, fluid_density], axis=0)
                print('add', points.shape, vel.shape, density.shape)

        if pos.shape[0]:
            fluid_output_path = os.path.join(output_dir,
                                             'fluid_{0:04d}'.format(step))
            
            # write_particles(fluid_output_path, pos, vel, density, options)
            if isinstance(pos, np.ndarray):
                write_particles(fluid_output_path, pos, vel, density, options)
            else:
                write_particles(fluid_output_path, pos.numpy(), vel.numpy(), density, options)

            inputs = (pos, vel, density, feats, box, box_normals)
            pos, vel = model(inputs)

        # remove out of bounds particles
        if step % 10 == 0:
            try:
                mask = pos[:, 1] > min_y
                print(step, 'num particles', pos.shape[0])
                print(type(pos))
                print(type(mask))
                print(type(density))
                print(type(vel))
                print("mask shape:", mask.shape)
                print("density shape:", density.shape)
                print("pos shape:", pos.shape)
                if np.count_nonzero(mask) < pos.shape[0]:
                    mask = mask.numpy()
                    density = density[mask]
                    pos = pos[mask]
                    vel = vel[mask]
            except Exception as e:
                print(e)
                print(mask)
                print("mask shape:", mask.shape)
                print("density shape:", density.shape)
                print("pos shape:", pos.shape)
                print("vel shape:", vel.shape)
                print("box shape:", box.shape)
                print("box_normals shape:",box_normals.shape)

    end_time = time.time()  
    print('Total time: ', end_time - start_time)
    print('average time: ', (end_time - start_time) / num_steps)


def main():
    parser = argparse.ArgumentParser(
        description=
        "Runs a fluid network on the given scene and saves the particle positions as npz sequence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("trainscript",
                        type=str,
                        help="The python training script.")
    parser.add_argument("--cfg",
                        type=str,
                        help="The path to the yaml config file")
    parser.add_argument('--gpu',
                        help='指定使用的GPU ID，例如：0,1,2',
                        type=str,
                        default='0')
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help=
        "The path to the .h5 network weights file for tensorflow ot the .pt weights file for torch."
    )
    parser.add_argument("--num_steps",
                        type=int,
                        default=250,
                        help="The number of simulation steps. Default is 250.")
    parser.add_argument("--scene",
                        type=str,
                        required=True,
                        help="A json file which describes the scene.")
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="The output directory for the particle data.")
    parser.add_argument("--write-ply",
                        action='store_true',
                        help="Export particle data also as .ply sequence")
    parser.add_argument("--write-bgeo",
                        action='store_true',
                        help="Export particle data also as .bgeo sequence")
    parser.add_argument("--device",
                        type=str,
                        default='cuda',
                        help="The device to use. Applies only for torch.")

    args = parser.parse_args()
    print(args)

    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    '''if args.trainscript is /path/to/my_script.py, then module_name is set to my_script
    this is train_network_tf or train_network_torch
    '''
    module_name = os.path.splitext(os.path.basename(args.trainscript))[0]
    print(module_name)
    '''adds the current directory to the module search path in Python. ensure that Python can find the module in the current directory that named module_name'''
    sys.path.append('.')
    '''use importlib.import_module dynamically imports module named module_name and assigns it to trainscript_module'''
    trainscript_module = importlib.import_module(module_name)

    with open(args.scene, 'r', encoding='utf-8') as f:
        scene = json.load(f)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    gpu_id = int(args.gpu)
    return run_sim_tf(trainscript_module, cfg, args.weights, scene,
                        args.num_steps, args.output, args, gpu=gpu_id)


if __name__ == '__main__':
    sys.exit(main())
