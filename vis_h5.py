import h5py
import numpy as np
import json
import os
import argparse
import random


def get_h5_structure(file_path):
    """
    读取h5文件的结构信息，包括组、数据集及其属性
    
    Args:
        file_path: h5文件路径
    
    Returns:
        h5文件结构的字典表示
    """
    def visit_item(name, obj):
        if isinstance(obj, h5py.Dataset):
            info = {
                'type': 'Dataset',
                'shape': obj.shape,
                'dtype': str(obj.dtype),
                'attrs': {k: str(v) for k, v in obj.attrs.items()}
            }
            # 获取数据样本（如果数据集不是太大）
            if len(obj.shape) > 0 and obj.shape[0] > 0:
                try:
                    if len(obj.shape) == 1:
                        sample = obj[0:min(5, obj.shape[0])] if random.random() < 0.5 else obj[-5:]
                    else:
                        sample = obj[0]
                    if isinstance(sample, np.ndarray):
                        info['sample'] = sample.tolist()
                    else:
                        info['sample'] = sample
                except Exception as e:
                    info['sample_error'] = str(e)
            structure[name] = info
        elif isinstance(obj, h5py.Group):
            structure[name] = {
                'type': 'Group',
                'attrs': {k: str(v) for k, v in obj.attrs.items()}
            }

    structure = {}
    with h5py.File(file_path, 'r') as f:
        f.visititems(visit_item)
        # 根组的属性
        structure['root_attrs'] = {k: str(v) for k, v in f.attrs.items()}
    
    return structure


def save_h5_info(file_path, output_dir=None, output_format='json'):
    """
    保存h5文件的信息到文件中
    
    Args:
        file_path: h5文件路径
        output_dir: 输出目录，如果为None则使用h5文件所在目录
        output_format: 输出格式，支持'json'和'txt'
    
    Returns:
        输出文件的路径
    """
    if output_dir is None:
        output_dir = os.path.dirname(file_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_name = os.path.basename(file_path)
    base_name = os.path.splitext(file_name)[0]
    
    structure = get_h5_structure(file_path)
    
    if output_format == 'json':
        output_path = os.path.join(output_dir, f"{base_name}_info.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(structure, f, indent=4)
    else:  # txt格式
        output_path = os.path.join(output_dir, f"{base_name}_info.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"H5文件信息: {file_path}\n")
            f.write("=" * 80 + "\n\n")
            
            # 根组属性
            f.write("根组属性:\n")
            for k, v in structure.get('root_attrs', {}).items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")
            
            # 遍历所有组和数据集
            for name, info in sorted(structure.items()):
                if name == 'root_attrs':
                    continue
                
                if info['type'] == 'Group':
                    f.write(f"组: {name}\n")
                    f.write("-" * 40 + "\n")
                    for k, v in info.get('attrs', {}).items():
                        f.write(f"  属性 {k}: {v}\n")
                else:  # Dataset
                    f.write(f"数据集: {name}\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"  形状: {info['shape']}\n")
                    f.write(f"  类型: {info['dtype']}\n")
                    
                    for k, v in info.get('attrs', {}).items():
                        f.write(f"  属性 {k}: {v}\n")
                    
                    if 'sample' in info:
                        f.write(f"  样本: {info['sample']}\n")
                    elif 'sample_error' in info:
                        f.write(f"  样本读取错误: {info['sample_error']}\n")
                
                f.write("\n")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='读取H5文件并保存其信息')
    parser.add_argument('file_path', help='H5文件路径')
    parser.add_argument('--output_dir', help='输出目录（默认为H5文件所在目录）', default=None)
    parser.add_argument('--format', choices=['json', 'txt'], default='json', help='输出格式（默认为json）')
    
    args = parser.parse_args()
    
    output_path = save_h5_info(args.file_path, args.output_dir, args.format)
    print(f"H5 file information saved to: {output_path}")


if __name__ == "__main__":
    main()

