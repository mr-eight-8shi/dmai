import os
import sys
import numpy as np


class MNISTDataLoader:
    """Class to load and normalize MNIST binary files, and provide filtered or full datasets."""

    def __init__(self, config=None):
        self.config = config
        self.data_dir = getattr(config, "dataset_dir", "dataset") if config else "dataset"
        
        self._X_train = None
        self._y_train = None
        self._X_test = None
        self._y_test = None

    def _ensure_train_loaded(self):
        """Load training data on-demand if not already loaded."""
        if self._X_train is None or self._y_train is None:
            img_path = os.path.join(self.data_dir, "train-images.idx3-ubyte")
            label_path = os.path.join(self.data_dir, "train-labels.idx1-ubyte")
            self._X_train, self._y_train = self._load_ubyte(img_path, label_path, "Training")

    def _ensure_test_loaded(self):
        """Load test data on-demand if not already loaded."""
        if self._X_test is None or self._y_test is None:
            img_path = os.path.join(self.data_dir, "t10k-images.idx3-ubyte")
            label_path = os.path.join(self.data_dir, "t10k-labels.idx1-ubyte")
            self._X_test, self._y_test = self._load_ubyte(img_path, label_path, "Test")

    def load_train_data_by_classes(self, class_min: int, class_max: int):
        """Return training data filtered by the specified class range."""
        self._ensure_train_loaded()
        mask = (self._y_train >= class_min) & (self._y_train <= class_max)
        return self._X_train[mask], self._y_train[mask]

    def load_test_data_by_classes(self, class_min: int, class_max: int):
        """Return test data filtered by the specified class range."""
        self._ensure_test_loaded()
        mask = (self._y_test >= class_min) & (self._y_test <= class_max)
        return self._X_test[mask], self._y_test[mask]

    def load_test_data(self):
        """Return all test data (classes 0-9)."""
        self._ensure_test_loaded()
        return self._X_test, self._y_test

    def _load_ubyte(self, img_path, label_path, dataset_type_name):
        if not os.path.exists(img_path) or not os.path.exists(label_path):
            print(f"[ERROR] Data file not found: {img_path} or {label_path}")
            print("Please ensure MNIST binary files are placed in the dataset directory.")
            sys.exit(1)

        print(f"Loading {dataset_type_name} dataset...")
        images = self._load_images(img_path)
        labels = self._load_labels(label_path)
        
        X = images
        y = labels
        
        print(f"Data loaded successfully (Total samples: {len(X)}, Shape: {X.shape[1]}x{X.shape[2]})")
        return X, y
        
    def _load_images(self, file_path):
        with open(file_path, "rb") as f:
            _ = f.read(4)  # Read magic number
            num_images = int.from_bytes(f.read(4), "big")
            rows = int.from_bytes(f.read(4), "big")
            cols = int.from_bytes(f.read(4), "big")
            data = np.frombuffer(f.read(), dtype=np.uint8)
            # Normalize pixel values to the range [0.0, 1.0]
            return data.reshape(num_images, rows, cols).astype(np.float32) / 255.0

    def _load_labels(self, file_path):
        with open(file_path, "rb") as f:
            _ = f.read(4)  # Read magic number
            _ = f.read(4)  # Read number of items
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.astype(np.int64)