More formal, organized documentation other than program comments. 

# Personal Notes
  Current Focuses:
  - Importing data, image data and labels -> testing/evaluation data
  - Look for any errors and debug
  - Specify paths for training data storage and output for highlighted images
  - work on documentation process and ensure deep understanding of all functions of the model

  Later Focuses:
   - Compile visualization task
   - Figure out how to evaluate accuracy 
   - Run test runs, go through any bugs and errors 

  Later Consideration:
   - implement frontend 

# Keras Built-In Methods
  - model.compile: configures model for training by specifying optimizer, loss function, and metrics to be used. 
  1) Optimizer - used to change attributes of neural networks such as weights and learning rate
    This model uses an Adam optimizer - used to update network weights based on training data, and is notable due to its adaptive learning rates that will make it optimal for identifying patterns and correcting error. 
  2) Loss Function - measures how well model's predictions match actual target values, in this case by measuring how well the model's predicted probabilities match ground truth (labels)
    This model uses Binary Cross-Entropy Loss - performs binary classification to determine stroke VS no stroke
  3) Metrics - specifies metrics to be tracked during training and evaluation.
    The metrics tracked by the model will be accuracy.

  - model.fit: initiates training based on given dataset. 

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

# Training
  - train_generator: instance of DataGenerator class.
  - model.compile 
  - model.fit
      -trains model using data generator (instance of DataGenerator class, train_generator) for 10 epochs
      -During each epoch, following steps are repeated:
        1) batch of input data and labels/ground truth and provided
        2) model makes predictions
        3) binary cross-entropy loss is computed (using comparisons between data and ground truth)
        4) using evaluation from binary cross-entropy, model's weights are updated
        5) accuracy is evaluated using metrics

# Creating Checkpoints
  - Checkpoint Directory: specifies where on local filesystem the model checkpoints will be saved
  - Checkpoint File Path: specifies naming convention and exact file path for checkpoint directory