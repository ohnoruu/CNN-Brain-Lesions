import os

os.environ['CUDA_PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2'
os.environ['CUDA_PATH_V11_2'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2'
os.environ['PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin;' + os.environ['PATH']
os.environ['PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp;' + os.environ['PATH']
os.environ['PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include;' + os.environ['PATH']
os.environ['PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\extras\CUPTI\lib64;' + os.environ['PATH']
os.environ['PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib\x64;' + os.environ['PATH']

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf

print("Tensorflow Version: ", tf.__version__)
print("Num GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))