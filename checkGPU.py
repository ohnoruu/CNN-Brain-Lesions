import tensorflow as tf
import os
print("Tensorflow Version: ", tf.__version__)
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
print("Num GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))