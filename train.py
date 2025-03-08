# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from config import CSV_FILE, IMG_DIR, DEVICE, NUM_CLASSES, REGRESSION_OUTPUT_SIZE, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, CLASSIFICATION_LOSS_WEIGHT, REGRESSION_LOSS_WEIGHT, MODEL_FOLDER, MODEL_NAME
from dataset import ColorDataset
import augmentation as aug
from model import MultiTaskColorModel

def get_dataloader():
    """
    Create and return a DataLoader combining original and augmented datasets.
    """
    original_dataset = ColorDataset(csv_file=CSV_FILE, img_dir=IMG_DIR, transform=aug.get_original_transform())
    augmented_dataset_1 = ColorDataset(csv_file=CSV_FILE, img_dir=IMG_DIR, transform=aug.get_augmented_transform_1())
    augmented_dataset_2 = ColorDataset(csv_file=CSV_FILE, img_dir=IMG_DIR, transform=aug.get_augmented_transform_2())
    
    # Combine datasets (you can adjust the combination as needed)
    combined_dataset = ConcatDataset([original_dataset, augmented_dataset_1, original_dataset, augmented_dataset_2])
    return DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True)

def train():
    """
    Trains the MultiTaskColorModel on the combined dataset and saves the trained model.
    """
    data_loader = get_dataloader()
    model = MultiTaskColorModel(num_classes=NUM_CLASSES, regression_output_size=REGRESSION_OUTPUT_SIZE).to(DEVICE)
    
    # Define loss functions
    criterion_class = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch")
        
        for images, _, color_codes, encoded_labels in pbar:
            images = images.to(DEVICE)
            encoded_labels = encoded_labels.to(DEVICE)
            color_codes = color_codes.to(DEVICE)
            
            optimizer.zero_grad()
            class_outputs, reg_outputs = model(images)
            
            loss_class = criterion_class(class_outputs, encoded_labels)
            loss_reg = criterion_reg(reg_outputs, color_codes)
            loss = CLASSIFICATION_LOSS_WEIGHT * loss_class + REGRESSION_LOSS_WEIGHT * loss_reg
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {running_loss/len(data_loader):.4f}")
    
    # Save the model (create folder if necessary)
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    save_path = os.path.join(MODEL_FOLDER, MODEL_NAME)
    print(f"Saving model to {save_path}")
    torch.save(model.state_dict(), save_path)
    
if __name__ == '__main__':
    train()
