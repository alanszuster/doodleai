import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dropout, BatchNormalization,
    GlobalAveragePooling2D, Dense, RandomRotation, RandomTranslation, RandomZoom
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
import os

os.makedirs("model", exist_ok=True)

X = np.load("dataset/processed/X_train.npy")
y = np.load("dataset/processed/y_train.npy")

print(f"Data range: {X.min():.3f} to {X.max():.3f}")
print(f"Data shape: {X.shape}")
print(f"Data type: {X.dtype}")

with open("model/classes.json", "r") as f:
    class_mapping = json.load(f)
num_classes = len(class_mapping)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def build_model(input_shape=(28, 28, 1), num_classes=20):
    model = Sequential([
        tf.keras.layers.Input(shape=input_shape),
        # Data augmentation - active only during training, disabled at inference
        RandomRotation(0.15),
        RandomTranslation(0.1, 0.1),
        RandomZoom(0.1),

        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.3),

        Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        GlobalAveragePooling2D(),

        Dense(512, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.5),
        Dense(256, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    return model


model = build_model(input_shape=(28, 28, 1), num_classes=num_classes)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

callbacks = [
    ModelCheckpoint(
        "model/best_model.keras",
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
