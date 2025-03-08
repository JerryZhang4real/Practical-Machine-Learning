# dataset.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import LabelEncoder

class ColorDataset(Dataset):
    """
    A custom dataset for loading images along with their associated labels and color codes.
    """
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with labels and filenames.
            img_dir (str): Directory containing image files.
            transform (callable, optional): Transformations to apply to each image.
        """
        self.image_labels = pd.read_csv(csv_file, encoding='utf-8')
        self.img_dir = img_dir
        self.transform = transform
        
        # Fit a LabelEncoder to encode class labels once
        self.label_encoder = LabelEncoder()
        self.image_labels['encoded_label'] = self.label_encoder.fit_transform(self.image_labels['label'])
    
    def __len__(self):
        return len(self.image_labels)
    
    def __getitem__(self, idx):
        # Construct image path and load image
        img_name = self.image_labels.iloc[idx]['filename']
        img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Retrieve original and encoded labels
        label = self.image_labels.iloc[idx]['label']
        encoded_label = self.image_labels.iloc[idx]['encoded_label']
        encoded_label = torch.tensor(encoded_label, dtype=torch.long)
        
        # Retrieve additional color code information
        color_code = self.image_labels.iloc[idx][['r', 'g', 'b', 'LRV']].values.astype('float32')
        color_code = torch.tensor(color_code, dtype=torch.float32)
        
        return image, label, color_code, encoded_label
