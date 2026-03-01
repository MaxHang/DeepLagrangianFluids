import tensorflow as tf
import sys
import os

print("="*50)
print("Environment Verification")
print("="*50)
print(f"Python Version: {sys.version}")
print(f"TensorFlow Version: {tf.__version__}")
try:
    print(f"Keras Version: {tf.keras.__version__}")
except:
    pass

print("\n--- GPU Information ---")
gpus = tf.config.list_physical_devices('GPU')
print(f"Num GPUs Available: {len(gpus)}")
for i, gpu in enumerate(gpus):
    print(f"  GPU {i}: {gpu}")

print("\n--- Build Information ---")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"Built with GPU support: {tf.test.is_built_with_gpu_support()}")

try:
    from tensorflow.python.platform import build_info as tf_build_info
    print("\n--- CUDA/cuDNN Versions ---")
    if hasattr(tf_build_info, 'build_info'):
        print(f"Cuda Version: {tf_build_info.build_info.get('cuda_version', 'N/A')}")
        print(f"Cudnn Version: {tf_build_info.build_info.get('cudnn_version', 'N/A')}")
except Exception as e:
    print(f"\nCould not retrieve detailed build info: {e}")

print("="*50)
