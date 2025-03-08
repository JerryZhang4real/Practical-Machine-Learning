# model.py
import torch.nn as nn
from torchvision import models

class MultiTaskColorModel(nn.Module):
    def __init__(self, num_classes, regression_output_size):
        """
        Initializes the multi-task model with a ResNet50 backbone.
        
        Args:
            num_classes (int): Number of classes for classification.
            regression_output_size (int): Number of outputs for regression.
        """
        super(MultiTaskColorModel, self).__init__()
        
        # Use pre-trained ResNet50 as the backbone and remove the final classification layer.
        self.backbone = models.resnet50(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        
        # Regression head
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
