import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization, Conv2D, Dense, Dropout,
    GlobalAveragePooling2D, MaxPooling2D, RandomRotation, RandomTranslation, RandomZoom
)
from tensorflow.keras.models import Sequential

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED_DIR = os.path.join(ROOT_DIR, 'dataset', 'processed')
MODEL_DIR = os.path.join(ROOT_DIR, 'model')

# Use all CPU cores
tf.config.threading.set_intra_op_parallelism_threads(12)
tf.config.threading.set_inter_op_parallelism_threads(12)

os.makedirs(MODEL_DIR, exist_ok=True)

X = np.load(os.path.join(PROCESSED_DIR, 'X_train.npy'))
y = np.load(os.path.join(PROCESSED_DIR, 'y_train.npy'))
X_val = np.load(os.path.join(PROCESSED_DIR, 'X_val.npy'))
y_val = np.load(os.path.join(PROCESSED_DIR, 'y_val.npy'))

print(f"Data range: {X.min():.3f} to {X.max():.3f}")
print(f"Data shape: {X.shape}")
print(f"Data type: {X.dtype}")

with open(os.path.join(MODEL_DIR, 'classes.json'), "r", encoding="utf-8") as f:
    class_mapping = json.load(f)
num_classes = len(class_mapping)

X_train, y_train = X, y


def build_model(input_shape=(28, 28, 1), n_classes=20):
    net = Sequential([
        tf.keras.layers.Input(shape=input_shape),
        # Data augmentation - active only during training, disabled at inference
        RandomRotation(0.15),
        RandomTranslation(0.1, 0.1),
        RandomZoom(0.1),

        # 28x28
        Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.25),

        # 14x14
        Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.3),

        # 7x7
        Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.3),

        # 3x3
        Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        GlobalAveragePooling2D(),
        Dropout(0.4),

        Dense(256, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.4),
        Dense(128, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ])
    return net


model = build_model(input_shape=(28, 28, 1), n_classes=num_classes)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

callbacks = [
    ModelCheckpoint(
        os.path.join(MODEL_DIR, 'best_model.keras'),
        save_best_only=True,
        monitor="val_accuracy",
        mode="max",
        verbose=1
    ),
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1)
]

print("Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=256,
    epochs=80,
    callbacks=callbacks,
    verbose=1
)

final_val_acc = max(history.history['val_accuracy'])
print(f"Best validation accuracy: {final_val_acc:.4f} ({final_val_acc * 100:.2f}%)")
