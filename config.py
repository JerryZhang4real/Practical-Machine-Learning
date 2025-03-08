# config.py
import torch

# === Data paths ===
CSV_FILE = r'D:\COSI149\color_with_label\label.csv'
IMG_DIR = r'D:\COSI149\color_with_label'

# === Training hyperparameters ===
BATCH_SIZE = 48
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

# === Loss weights ===
CLASSIFICATION_LOSS_WEIGHT = 0.1
REGRESSION_LOSS_WEIGHT = 0.9

# === Model parameters ===
NUM_CLASSES = 8
REGRESSION_OUTPUT_SIZE = 4

# === Device configuration ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Model saving parameters ===
MODEL_FOLDER = 'models'
MODEL_ARCHI = 'ResNet50'
MODEL_VERSION = 'v1'
MODEL_NAME = MODEL_ARCHI + '_' + str(NUM_EPOCHS) + '_' + str(CLASSIFICATION_LOSS_WEIGHT) + \
    '_' + str(REGRESSION_LOSS_WEIGHT) + '_' + MODEL_VERSION + '.pth'

# config into
CONFIG_INFO = \
        f"Model: {MODEL_NAME}\n" + \
        f"Learning Rate: {LEARNING_RATE}\n" + \
        f"Epochs: {NUM_EPOCHS}\n" + \
        f"Classification Loss Weight: {CLASSIFICATION_LOSS_WEIGHT}" + "REGRESSION_LOSS_WEIGHT: {REGRESSION_LOSS_WEIGHT}\n" + \
        f"\n"
