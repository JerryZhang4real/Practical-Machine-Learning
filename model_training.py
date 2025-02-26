# %%
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# %%
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder

# Data loading
class ColorDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.image_labels = pd.read_csv(csv_file, encoding='utf-8')
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_labels)
    
    def __getitem__(self, idx):
        img_name = self.image_labels.iloc[idx]['filename']
        img_path = os.path.join(self.img_dir, img_name+'.jpg')
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)


        class_label = self.image_labels.iloc[idx]['label']

        label_encoder = LabelEncoder()
        self.image_labels['encoded_label'] = label_encoder.fit_transform(self.image_labels['label'])

        class_encoded_label = self.image_labels.iloc[idx]['encoded_label']
        class_encoded_label = torch.tensor(class_encoded_label, dtype=torch.long)

        color_code = self.image_labels.iloc[idx][['r', 'g', 'b', 'LRV']].values.astype('float32')
        color_code = torch.tensor(color_code, dtype=torch.float32)

        return image, class_label, color_code, class_encoded_label
    




# %%
# use torchvision to do data augmentation
import torchvision.transforms.v2 as transforms


original_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define a transform to augment the data
augmented_transform_1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

augmented_transform_2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomPerspective(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

csv_file = r'D:\COSI149\color_with_label\label.csv'
if not os.path.isfile(csv_file):
    raise FileNotFoundError(f"The file {csv_file} does not exist. Please provide the correct path to the file.")

img_dir = r'D:\COSI149\color_with_label'  # Update this path to the correct directory containing images

original_dataset = ColorDataset(csv_file=csv_file, img_dir=img_dir, transform=original_transform)

# Visualize the original dataset
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1, 5, figsize=(15, 3))

# for i in range(5):
#     image, class_label, color_code, class_encoded_label = original_dataset[i*7]
#     ax[i].imshow(transforms.ToPILImage()(image))
#     ax[i].set_title(f'Label: {class_label}\nEncoded_Label: {class_encoded_label}\nColor: {color_code.numpy()}')
#     ax[i].axis('off')

# plt.show()
# create the dataset with the new transform

augmented_dataset_1 = ColorDataset(csv_file=csv_file, img_dir=img_dir, transform=augmented_transform_1)
# augmented_dataset_2 = ColorDataset(csv_file=csv_file, img_dir=img_dir, transform=augmented_transform_2)

# Visualize the augmented data
# fig, ax = plt.subplots(1, 5, figsize=(15, 3))

# for i in range(5):
#     image, class_label, color_code, class_encoded_label = augmented_dataset[i*7]
#     ax[i].imshow(transforms.ToPILImage()(image))
#     ax[i].set_title(f'Label: {class_label}\nEncoded_Label: {class_encoded_label}\nColor: {color_code.numpy()}')
#     ax[i].axis('off')

# plt.show()  

# %%
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

# conbined_dataset = ConcatDataset([original_dataset, augmented_dataset_1, augmented_dataset_2, original_dataset, augmented_dataset_2])
# conbined_dataset = ConcatDataset([original_dataset, augmented_dataset_1, augmented_dataset_2])
conbined_dataset = ConcatDataset([original_dataset, augmented_dataset_1])
data_loader = DataLoader(conbined_dataset, batch_size=48, shuffle=True)

# # Visualize the data   
# fig, ax = plt.subplots(1, 5, figsize=(15, 3))

# for i, (images, class_labels, color_codes, class_encoded_label) in enumerate(data_loader):
#     if i >= 3:
#         break
#     ax[i].imshow(transforms.ToPILImage()(images[0]))
#     ax[i].set_title(f'Label: {class_labels[0]}\nColor: {color_codes[0].numpy()}')
#     ax[i].axis('off')

# plt.show()

data_loader.__len__()

# %%
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
classsification_loss_weight = 0.1
regression_loss_weight = 1.0

#Optimizer
learning_rate = 0.001
model = MultiTaskColorModel(num_classes=8, regression_output_size=4)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# %%
# Training the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels, color_codes, encoded_labels in data_loader:
        images = images.to(device)
        labels = encoded_labels.to(device)
        color_codes = color_codes.to(device)

        optimizer.zero_grad()

        class_outputs, reg_outputs = model(images)

        loss_class = criterion_class(class_outputs, labels)
        loss_reg = criterion_reg(reg_outputs, color_codes)

        # Total loss is a weighted sum of classification and regression losses
        loss = classsification_loss_weight * loss_class + regression_loss_weight * loss_reg

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(data_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# %%
# save the model
torch.save(model.state_dict(), 'ResNet_152_50_0.1_1.0_v2.pth')




# %%
