# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 19:25:18 2025

@author: Gavin
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Data loading and preprocessing - this is important!
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL Image (0-255) to PyTorch tensor (0.0-1.0)
    # Note: ToTensor() automatically normalizes to 0-1 range, so we're good to go!
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define a Convolutional Neural Network
class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        
        # TODO: Define convolutional layers
        # Hint: Use nn.Conv2d, nn.MaxPool2d, nn.ReLU
        # Input: 1 channel (grayscale), 28x28 pixels
        
        # TODO: Define fully connected layers
        # You'll need to calculate the size after conv/pooling layers
        
        pass
    
    def forward(self, x):
        # TODO: Implement the forward pass
        # Apply conv layers, pooling, activation functions
        # Flatten before fully connected layers
        # Return final output
        
        pass

# TODO: Create model, loss function, and optimizer
model = MNISTCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# TODO: Implement training loop (similar to before)
def train_cnn(num_epochs=5):
    # TODO: Train the CNN and track loss/accuracy
    pass

# TODO: Implement testing function
def test_cnn():
    # TODO: Evaluate the CNN on test data
    pass

# TODO: Train and test your CNN
print("Training CNN...")
# train_cnn()
# test_cnn()

# TODO: | CHALLENGE | Visualize the learned convolutional filters
def visualize_conv_filters():
    # TODO: Extract and plot the first layer's convolutional filters
    pass

# visualize_conv_filters()