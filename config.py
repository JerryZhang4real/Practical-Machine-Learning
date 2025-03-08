# config.py
import torch

# === Data paths ===
CSV_FILE = r'D:\COSI149\color_with_label\label.csv'
IMG_DIR = r'D:\COSI149\color_with_label'

# === Model saving parameters ===
MODEL_FOLDER = 'models'
MODEL_NAME = 'ResNet50_100_0.1_0.9_v1.pth'

# === Training hyperparameters ===
BATCH_SIZE = 48
LEARNING_RATE = 0.001
NUM_EPOCHS = 100

# === Loss weights ===
CLASSIFICATION_LOSS_WEIGHT = 0.1
REGRESSION_LOSS_WEIGHT = 0.9

# === Model parameters ===
NUM_CLASSES = 8
REGRESSION_OUTPUT_SIZE = 4

# === Device configuration ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# config into
CONFIG_INFO = \
        f"Model: {MODEL_NAME}\n" + \
        f"Learning Rate: {LEARNING_RATE}\n" + \
        f"Epochs: {NUM_EPOCHS}\n" + \
        f"Classification Loss Weight: {CLASSIFICATION_LOSS_WEIGHT}" + "REGRESSION_LOSS_WEIGHT: {REGRESSION_LOSS_WEIGHT}\n" + \
        f"\n"
