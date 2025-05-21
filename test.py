# dataset_dir = "/datasets/cconv/ours_default_data"
# from glob import glob
# import os
# # val_files = sorted(glob(os.path.join(dataset_dir, 'test', '*.zst')))
# val_files = sorted(glob(os.path.join(dataset_dir, 'train', '*.zst')))
# print(val_files)

import h5py as h5py
# file = "/workspace/xyh_synology/graduate/datasets/nomix-fluid-cconv/valid/density_1000_2000_box_1_cd_1.0_cf_0.0.h5"
# with h5py.File(file, 'r') as f:
#     print(f.keys())
    
#     print(f.attrs)
#     for k, v in f.attrs.items():
#         print(k, v)
#     print(f["frames/1/rest_density"][0:10])


#
a = range(0,401,400)
print(a)
for step in range(1200):
    if step in a:
        print(step)