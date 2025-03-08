# evaluate.py
import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from utils import log_performance
from config import CSV_FILE, IMG_DIR, MODEL_FOLDER, MODEL_NAME, BATCH_SIZE, DEVICE, NUM_CLASSES, REGRESSION_OUTPUT_SIZE
from dataset import ColorDataset
import augmentation as aug
from model import MultiTaskColorModel

def evaluate():
    """
    Loads the trained model and evaluates it on the test/validation dataset.
    """
    dataset = ColorDataset(csv_file=CSV_FILE, img_dir=IMG_DIR, transform=aug.get_original_transform())
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = MultiTaskColorModel(num_classes=NUM_CLASSES, regression_output_size=REGRESSION_OUTPUT_SIZE).to(DEVICE)
    model_path = os.path.join(MODEL_FOLDER, MODEL_NAME)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    total_correct = 0
    total_samples = 0
    total_reg_loss = 0.0
    all_labels = []
    all_predicted = []
    
    criterion_reg = torch.nn.MSELoss()
    
    with torch.no_grad():
        for images, labels, color_codes, encoded_labels in dataloader:
            images = images.to(DEVICE)
            encoded_labels = encoded_labels.to(DEVICE)
            color_codes = color_codes.to(DEVICE)
            
            class_outputs, reg_outputs = model(images)
            _, predicted = torch.max(class_outputs, 1)
            
            total_samples += encoded_labels.size(0)
            total_correct += (predicted == encoded_labels).sum().item()
            all_labels += encoded_labels.cpu().tolist()
            all_predicted += predicted.cpu().tolist()
            
            loss_reg = criterion_reg(reg_outputs, color_codes)
            total_reg_loss += loss_reg.item()
    
    accuracy = total_correct / total_samples
    avg_reg_loss = total_reg_loss / len(dataloader)
    print(f"Validation Accuracy: {accuracy*100:.2f}%")
    print(f"Average Regression Loss: {avg_reg_loss:.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_predicted)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    log_performance(accuracy=accuracy, reg_loss=avg_reg_loss) 
if __name__ == '__main__':
    evaluate()
