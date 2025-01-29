import tensorflow as tf
from tensorflow.keras.layers import Layer

class LearnableAugmentation(Layer):
    def __init__(self, angle_range=(-10, 10), **kwargs):
        super(LearnableAugmentation, self).__init__(**kwargs)
        self.angle_range = angle_range

    def build(self, input_shape):
        # Create a learnable weight for the angle of rotation
        self.angle = self.add_weight(
            shape=(1,),
            initializer='zeros',
            trainable=True,
            name='angle'
        )
        super(LearnableAugmentation, self).build(input_shape)

    def call(self, inputs):
        # Ensure inputs are a tuple of images and labels, even if passed as a list
        if isinstance(inputs, list):
            images, labels = inputs[0], inputs[1]
        else:
            images, labels = inputs

        # Ensure angle stays within the specified range
        clipped_angle = tf.clip_by_value(self.angle, self.angle_range[0], self.angle_range[1])
        radian_angle = clipped_angle * tf.constant(3.14159 / 180, dtype=tf.float32)

        # Normalize the angle for the transformation
        normalized_angle = tf.squeeze(radian_angle)

        # Ensure the input images have 4 dimensions (batch_size, height, width, channels)
        images = self._ensure_4d(images)
        labels = self._ensure_4d(labels)

        # Rotate images and labels
        rotated_images = self._rotate_with_tf(images, normalized_angle)
        rotated_labels = self._rotate_with_tf(labels, normalized_angle, is_mask=True)

        return rotated_images, rotated_labels

    def _rotate_with_tf(self, images, angle, is_mask=False):
        # Rotation logic: we could apply some rotation using standard matrix transformations
        # For simplicity, we are resizing and rotating the images here

        # Define the rotation matrix and apply the transformation
        cos_theta = tf.cos(angle)
        sin_theta = tf.sin(angle)

        # Create the affine rotation matrix
        rotation_matrix = tf.stack([cos_theta, -sin_theta, sin_theta, cos_theta], axis=0)
        rotation_matrix = tf.reshape(rotation_matrix, [2, 2])

        # Apply rotation using tf.image (after matrix multiplication)
        rotated_images = tf.image.resize(images, (224, 224))
        
        if is_mask:
            rotated_images = tf.image.resize(images, (224, 224), method='nearest')  # Preserve binary values
        
        return rotated_images

    def _ensure_4d(self, tensor):
        """Ensure the input tensor has 4 dimensions (batch_size, height, width, channels)"""
        if len(tensor.shape) == 3:  # Single image (height, width, channels)
            tensor = tf.expand_dims(tensor, 0)  # Add batch dimension
        return tensor

    def get_config(self):
        config = super(LearnableAugmentation, self).get_config()
        config.update({'angle_range': self.angle_range})
        return config
