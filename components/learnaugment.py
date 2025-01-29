import tensorflow as tf
from tensorflow.keras.layers import Layer # class for custom layers
import keras_cv

# Augmentation is currently rotation-based. 
class LearnableAugmentation(Layer):
    def __init__(self, angle_range=(-10,10), **kwargs):
        # set up subclass with specific angle range
        super(LearnableAugmentation, self).__init__(**kwargs)
        self.angle_range = angle_range

    def build(self, input_shape):
        # create learnable angloe parameter with initial value 0
        # angle will be adjusted during training to minimize loss

        # self.add_weight is provided by Tensorflow from the Layer class used to define parameters for custom layers
        # specifies shape, initializer, trainability and name
        self.angle = self.add_weight(
            shape=(1,),
            initializer='zeros',
            trainable=True,
            name='angle'
        )
        super(LearnableAugmentation, self).build(input_shape)

    def call(self, inputs):
        images, labels = inputs

        # Ensure angle stays within range
        clipped_angle = tf.clip_by_value(self.angle, self.angle_range[0], self.angle_range[1])
        radian_angle = clipped_angle * tf.constant(3.14159 / 180, dtype=tf.float32)

        # Normalize angle for keras_cv rotation factor (expects fraction of 360)
        normalized_factor = radian_angle / (2 * tf.constant(3.14159, dtype=tf.float32))

        # Rotate images normally (uses bilinear interpolation)
        rotated_images = keras_cv.layers.RandomRotation(factor=(normalized_factor))(images)

        # Rotate labels using nearest-neighbor interpolation to preserve binary values
        rotated_labels = keras_cv.layers.RandomRotation(factor=(normalized_factor), interpolation="nearest")(labels)

        return rotated_images, rotated_labels
        
    def get_config(self):
        # ensures customized layer can be saved and loaded during model serialization
        config = super(LearnableAugmentation, self).get_config()
        config.update({'angle_range': self.angle_range})
        return config