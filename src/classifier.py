import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image


class DrawingClassifier:
    def __init__(self):
        self.model = None
        self.classes = []
        self.load_classes()

    def load_classes(self):
        classes_path = 'model/classes.json'

        if not os.path.exists(classes_path):
            raise FileNotFoundError(
                f"Required file {classes_path} not found. "
                "Please ensure classes.json exists in the model directory."
            )

        try:
            with open(classes_path, 'r', encoding='utf-8') as f:
                classes_dict = json.load(f)
            self.classes = [classes_dict[str(i)] for i in range(len(classes_dict))]
            print(f"Loaded {len(self.classes)} classes from {classes_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading classes from {classes_path}: {e}") from e

    def create_simple_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(self.classes), activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def preprocess_image(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')

        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        image = image.convert('L')
        img_array = np.array(image)
        img_array = 255 - img_array
        img_array = img_array.astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        return img_array

    def load_model(self):
        model_paths = [
            os.getenv('MODEL_PATH', 'model/best_model.keras'),
            'model/drawing_model.keras',
            'model/best_model.keras'
        ]

        model_loaded = False
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    print(f"Loading model from {model_path}...")
                    self.model = tf.keras.models.load_model(model_path)
                    print(f"Model loaded successfully from {model_path}")
                    model_loaded = True
                    break
                except Exception as e:  # pylint: disable=broad-exception-caught
                    print(f"Failed to load model from {model_path}: {e}")
                    continue

        if not model_loaded:
            print("No pre-trained model found. Creating new model...")
            self.model = self.create_simple_model()
            print("Model created with random weights. Requires training for accuracy.")

    def predict(self, image):
        if self.model is None:
            return [{'class': 'error', 'confidence': 0.0}]

        try:
            processed_image = self.preprocess_image(image)
            predictions = self.model.predict(processed_image, verbose=0)
            top_indices = np.argsort(predictions[0])[::-1][:3]

            results = []
            for idx in top_indices:
                confidence = float(predictions[0][idx])
                class_name = self.classes[idx] if idx < len(self.classes) else f"class_{idx}"
                results.append({
                    'class': class_name,
                    'confidence': round(confidence * 100, 1)
                })

            return results

        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Error during prediction: {e}")
            return [{'class': 'error', 'confidence': 0.0}]

    def save_model(self, path='model/drawing_model.keras'):
        if self.model is not None:
            self.model.save(path)
            print(f"Model saved to {path}")
        else:
            print("No model to save. Train the model first.")
