import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


# Data loading
class ColorDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):

        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        img_name = self.data_frame.iloc[idx]['filename']
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        class_label = int(self.data_frame.iloc[idx]['label'])

        color_code = self.data_frame.iloc[idx][['r', 'g', 'b']].values.asatype('float32')
        color_code = torch.tensor(color_code, dtype=torch.float32)

        return image, class_label, color_code

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Using ImageNet normalization as an example; adjust as needed
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Model Architecture
import torch.nn as nn
from torchvision import models

class MultiTaskColorModel(nn.Module):
    def __init__(self, num_classes, regression_output_size):
        super(MultiTaskColorModel, self).__init__()

        #Use a pre-trained ResNet model as the backbone
        self.backbone = models.resnet152(pretrained=True)
        num_features = self.backbone.fc.in_features

        #Remove the final classification layer
        self.backbone.fc = nn.Identity()

        #Add a new classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

        #Add a new regression head
        self.regressor = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, regression_output_size)
        )


    def forward(self, x):
        features = self.backbone(x)
        class_out = self.classifier(features)
        reg_out = self.regressor(features)


        return class_out, reg_out
        
# Loss function and Opatimizer
import torch.optim as optim

criterion_class = nn.CrossEntropyLoss()
criterion_reg = nn.MSELoss()

# Hyperparameters for balancing the regerssion loss
regression_loss_weight = 1.0

#Optimizer
learning_rate = 0.001
model = MultiTaskColorModel(num_classes=60)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training loop
from torch.utils.data import DataLoader

dataset = ColorDataset(csv_file='data.csv', root_dir='color_with_label', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels, color_codes in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        color_codes = color_codes.to(device)

        optimizer.zero_grad()

        class_outputs, reg_outputs = model(images)

        loss_class = criterion_class(class_outputs, labels)
        loss_reg = criterion_reg(reg_outputs, color_codes)

        # Total loss is a weighted sum of classification and regression losses
        loss = loss_class + regression_loss_weight * loss_reg

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


# Validation and Evaluation
model.eval()
total_correct = 0
total_samples = 0
total_reg_loss = 0.0

with torch.no_gradJ():
    for images, labels, color_codes in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        color_codes = color_codes.to(device)

        class_outputs, reg_outputs = model(images)

        # Classifacation accuracy
        _, predicted = torch.max(class_outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

        # Regression loss
        loss_reg = criterion_reg(reg_outputs, color_codes)
        total_reg_loss += loss_reg.item()

    accuracy = total_correct / total_samples
    avg_reg_loss = total_reg_loss / len(dataloader)
    print(f"Validation Accuracy: {accuracy*100:.2f}%")
    print(f"Average Regression Loss: {avg_reg_loss:.4f}")


# Inference
new_image_path = 'path_to_new_image.jpg'
image = Image.open(new_image_path).convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    class_output, reg_output = model(image_tensor)
    predicted_class = torch.argmax(class_output, dim=1).item()
    predicted_color_code = reg_output.squeeze().cpu().numpy()

print(f"Predicted Class: {predicted_class}")
print(f"Predicted Color Code: {predicted_color_code}")