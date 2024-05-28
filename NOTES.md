More formal, organized documentation other than program comments. 
# DataGenerator Class
# Classification
1) Retrieving inputs
      inputs = Input(shape=input_shape)
   - creates an input layer for the model with the specified input shape
   - defines depth, height, width, and channels (colors)
  
2) Applying functions (round 1) 
     x = Conv3D(32, kernel_size=(3,3,3), activation='sigmoid', padding='same')(inputs)
   - creates a 3D Convolutional Layer on the input data
   - 32 - number of filters (output channels) 
   - kernel-size=(3,3,3) - defines the size of a 3D convolutional kernel: a small matrix that is convolved with an input image, used to extract features
   - activation = 'sigmoid' - specifies activation function of the layer. sigmoid maps input values from a range between 1 and 0 (used for classification)
   - padding = 'same' - pads the input such as the output has the same dimensions as the input
  
     x = BatchNormalization()(x)
   - normalizes activations of the previous layer
   - rescales input values to a common range that is easier for the machine to recognize
