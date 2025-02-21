from tensorflow.keras.utils import plot_model 
from components.modelfunctions import segmentation

model = segmentation((224, 224, 26, 1))
plot_model(model, to_file='model.png', show_shapes=True, rankdir="LR")
