import tensorflow as tf
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
print("Num GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))