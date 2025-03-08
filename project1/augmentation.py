# augmentation.py
from torchvision import transforms

def get_original_transform():
    """
    Returns a transform pipeline for the original (non-augmented) images.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

def get_augmented_transform_1():
    """
    Returns a transform pipeline with horizontal flip and slight rotation.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

def get_augmented_transform_2():
    """
    Returns a transform pipeline with vertical flip, larger rotation, and perspective distortion.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomPerspective(),
        transforms.ToTensor(),
    ])
