# ============================================================
# CS731 Emotion Recognition - Custom Dataset Loader
# Loads images from folder structure where each subfolder
# represents one emotion class (e.g. happy/, sad/, angry/)
# Used by train.py to create training and test datasets
# ============================================================

import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    """
    Custom PyTorch Dataset for loading emotion images.
    
    Expected folder structure:
        root_dir/
            anger/
                image001.jpg
                image002.jpg
            happy/
                image001.jpg
            sad/
                ...
    
    Each subfolder name becomes a class label.
    Labels are returned as one-hot encoded tensors.
    """

    def __init__(self, root_dir, transform=None):
        """
        Initialise the dataset by scanning the root directory.

        Args:
            root_dir (str): Root directory containing emotion subfolders
            transform (callable, optional): Image transforms to apply (resize, normalize etc.)
        """
        self.root_dir = root_dir
        self.transform = transform

        # Each subdirectory name is treated as a class label
        # os.listdir returns in alphabetical order — must stay consistent
        # between train and test datasets so labels match correctly
        self.classes = sorted(os.listdir(root_dir))
        self.num_classes = len(self.classes)

        # Map each class name to a numeric index
        # e.g. {'anger': 0, 'contempt': 1, 'disgust': 2, ...}
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Recursively find all image file paths in all subfolders
        self.image_paths = self._find_images()

        print(f"Loaded {len(self.image_paths)} images from {self.num_classes} classes.")
        print(f"Classes: {self.classes}")

    def _find_images(self):
        """
        Recursively scan root_dir and collect all image file paths.

        Returns:
            list: List of full file paths for all images found
        """
        # Supported image file extensions
        extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

        # Walk through all subdirectories and collect image paths
        return [
            os.path.join(root, name)
            for root, _, files in os.walk(self.root_dir)
            for name in files
            if name.lower().endswith(extensions)
        ]

    def __len__(self):
        """
        Return total number of images in dataset.
        Required by PyTorch DataLoader.

        Returns:
            int: Total number of image samples
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Load and return one image and its label at the given index.
        Required by PyTorch DataLoader — called automatically during training.

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            tuple: (image_tensor, label_tensor)
                   - image_tensor: Transformed image as PyTorch tensor [C, H, W]
                   - label_tensor: One-hot encoded label e.g. [0, 0, 0, 1, 0, 0, 0, 0]
        """
        # Get full path to this image
        image_path = self.image_paths[idx]

        # Determine class from parent folder name
        # e.g. "train_images/happy/img001.jpg" → class_name = "happy"
        class_name = os.path.basename(os.path.dirname(image_path))
        class_idx  = self.class_to_idx[class_name]

        # Load image and convert to RGB (handles grayscale and RGBA images too)
        image = Image.open(image_path).convert('RGB')

        # Apply transforms if provided (resize, normalize, augmentation etc.)
        if self.transform:
            image = self.transform(image)

        # Create one-hot encoded label tensor
        # e.g. class_idx=3 with 8 classes → [0, 0, 0, 1, 0, 0, 0, 0]
        label = torch.zeros(self.num_classes)
        label[class_idx] = 1

        return image, label
