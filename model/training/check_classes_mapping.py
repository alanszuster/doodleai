import os

import tensorflow as tf

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
model = tf.keras.models.load_model(os.path.join(ROOT_DIR, 'model', 'artifacts', 'best_model.keras'))
output_layer = model.layers[-1]
print("Number of output classes:", output_layer.units)
