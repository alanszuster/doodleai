import tensorflow as tf

model = tf.keras.models.load_model('model/best_model.keras')
output_layer = model.layers[-1]
print("Number of output classes:", output_layer.units)
