import os
import sys
import json
import shutil
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED_DIR = os.path.join(ROOT_DIR, 'dataset', 'processed')
QUICKDRAW_DIR = os.path.join(ROOT_DIR, 'dataset', 'quickdraw')
MODEL_DIR = os.path.join(ROOT_DIR, 'model')
os.makedirs(PROCESSED_DIR, exist_ok=True)

CLASSES = [
    # Animals
    'bear', 'bee', 'butterfly', 'cat', 'cow', 'crab', 'deer', 'dog',
    'dolphin', 'duck', 'elephant', 'fish', 'flamingo', 'frog', 'giraffe',
    'hedgehog', 'horse', 'kangaroo', 'lion', 'monkey', 'octopus', 'owl',
    'panda', 'penguin', 'pig', 'rabbit', 'shark', 'sheep', 'snake',
    'spider', 'tiger', 'whale', 'zebra',
    # Food
    'apple', 'banana', 'birthday cake', 'bread', 'carrot', 'cookie',
    'donut', 'grapes', 'hamburger', 'hot dog', 'ice cream', 'lemon',
    'mushroom', 'pear', 'pineapple', 'pizza', 'strawberry', 'watermelon',
    # Vehicles
    'airplane', 'bicycle', 'bus', 'car', 'firetruck', 'helicopter',
    'motorbike', 'rocket', 'sailboat', 'submarine', 'train', 'truck',
    # Objects
    'backpack', 'book', 'camera', 'chair', 'clock', 'computer', 'cup',
    'drums', 'fork', 'guitar', 'hammer', 'hat', 'key', 'knife', 'lamp',
    'microphone', 'pencil', 'piano', 'scissors', 'shoe', 'sword', 'umbrella',
    # Nature
    'cloud', 'fire', 'flower', 'leaf', 'lightning', 'moon', 'mountain',
    'rainbow', 'snowflake', 'star', 'sun', 'tree',
    # Buildings
    'bridge', 'castle', 'door', 'fence', 'house', 'lighthouse', 'windmill',
    # Body
    'ear', 'eye', 'face', 'hand', 'nose', 'tooth',
    # Misc
    'circle', 'crown', 'diamond', 'heart', 'hot air balloon', 'lollipop',
    'skull', 'stop sign', 'tornado', 'trophy',
]

BASE_URL = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'


def download_data(classes, data_dir=QUICKDRAW_DIR):
    os.makedirs(data_dir, exist_ok=True)
    for class_name in classes:
        file_name = f"{class_name}.npy"
        path = os.path.join(data_dir, file_name)
        if os.path.exists(path):
            print(f"Already exists: {file_name}")
            continue
        url = BASE_URL + class_name.replace(' ', '%20') + ".npy"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(path, 'wb') as f:
                f.write(r.content)
            print(f"Downloaded: {file_name}")
        except Exception as e:
            print(f"Failed to download {class_name}: {e}")


def load_data(classes, max_samples_per_class=15000):
    X, y, available_classes = [], [], []
    for class_name in classes:
        file_path = os.path.join(QUICKDRAW_DIR, f"{class_name}.npy")
        if not os.path.exists(file_path):
            print(f"Missing file: {class_name}")
            continue
        data = np.load(file_path)
        if data.shape[0] > max_samples_per_class:
            indices = np.random.choice(data.shape[0], max_samples_per_class, replace=False)
            data = data[indices]
        label_idx = len(available_classes)
        X.append(data)
        y.extend([label_idx] * data.shape[0])
        available_classes.append(class_name)
        print(f"Loaded {data.shape[0]} samples for '{class_name}'")
    if not X:
        raise RuntimeError("No data loaded. Check download step.")
    X = np.concatenate(X, axis=0).reshape(-1, 28, 28, 1).astype(np.float32) / 255.0
    y = np.array(y)
    return X, y, available_classes


def visualize_samples(X, y, classes, samples_per_class=5):
    fig, axes = plt.subplots(len(classes), samples_per_class, figsize=(samples_per_class * 2, len(classes) * 2))
    for class_idx, class_name in enumerate(classes):
        indices = np.where(y == class_idx)[0]
        samples = np.random.choice(indices, samples_per_class, replace=False)
        for i, idx in enumerate(samples):
            ax = axes[class_idx, i]
            ax.imshow(X[idx].squeeze(), cmap='gray')
            ax.axis('off')
            if i == 0:
                ax.set_title(class_name, fontsize=10)
    plt.tight_layout()
    output_path = os.path.join(PROCESSED_DIR, "sample_drawings.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved sample visualization to: {output_path}")


def split_and_save(X, y):
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=42)
    np.save(os.path.join(PROCESSED_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(PROCESSED_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(PROCESSED_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DIR, 'y_val.npy'), y_val)
    np.save(os.path.join(PROCESSED_DIR, 'y_test.npy'), y_test)
    print(f"Saved datasets: {X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test")


def save_class_mappings(classes):
    os.makedirs(MODEL_DIR, exist_ok=True)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    idx_to_class = {i: cls for i, cls in enumerate(classes)}
    with open(os.path.join(PROCESSED_DIR, 'class_name_to_index.json'), 'w') as f:
        json.dump(class_to_idx, f, indent=2)
    with open(os.path.join(PROCESSED_DIR, 'index_to_class_name.json'), 'w') as f:
        json.dump(idx_to_class, f, indent=2)
    shutil.copyfile(
        os.path.join(PROCESSED_DIR, 'index_to_class_name.json'),
        os.path.join(MODEL_DIR, 'classes.json')
    )
    print("Saved class mappings")


def main():
    print("Preparing QuickDraw dataset...")
    download_data(CLASSES)
    X, y, available_classes = load_data(CLASSES)
    visualize_samples(X, y, available_classes)
    split_and_save(X, y)
    save_class_mappings(available_classes)
    print("Done. Run scripts/train_model.py to train the model.")


if __name__ == "__main__":
    main()
