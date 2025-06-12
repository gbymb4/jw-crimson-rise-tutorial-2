# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 19:11:24 2025

@author: Gavin
"""

import torch

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# Define transforms for preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL Image (0-255) to PyTorch tensor (0.0-1.0)
    # Note: ToTensor() automatically normalizes to 0-1 range, so we're good to go!
])

# Download and load MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Let's look at some sample images
def show_sample_images():
    # Get a batch of training data
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    
    # Plot first 8 images
    fig, axes = plt.subplots(2, 4, figsize=(10, 6))
    for i in range(8):
        row = i // 4
        col = i % 4
        axes[row, col].imshow(images[i].squeeze(), cmap='gray')
        axes[row, col].set_title(f'Label: {labels[i].item()}')
        axes[row, col].axis('off')
    
    plt.suptitle('Sample MNIST Images')
    plt.tight_layout()
    plt.show()

show_sample_images()

# Define a simple neural network
class MNISTNet(nn.Module):
    
    def __init__(self):
        super(MNISTNet, self).__init__()
        # Flatten 28x28 images to 784 dimensional vector
        self.flatten = nn.Flatten()
        
        # Define the network layers
        self.fc1 = nn.Linear(28 * 28, 128)  # First hidden layer
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)       # Second hidden layer  
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)        # Output layer (10 classes for digits 0-9)
    
    
    
    def forward(self, x):
        x = self.flatten(x)  # Convert 28x28 image to 784-length vector
        x = self.fc1(x)      # First layer
        x = self.relu1(x)    # Activation function
        x = self.fc2(x)      # Second layer
        x = self.relu2(x)    # Activation function  
        x = self.fc3(x)      # Output layer
        return x



# Create the model
model = MNISTNet()
print("Model Architecture:")
print(model)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(num_epochs=5):
    model.train()
    train_losses = []
    train_accuracies = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}, '
                      f'Loss: {loss.item():.4f}')
        
        # Calculate epoch statistics
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Average Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    
    return train_losses, train_accuracies



# Test function
def test_model():
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f} ({100 * accuracy:.2f}%)')
    return accuracy



# Train the model
print("Starting training...")
train_losses, train_accuracies = train_model(num_epochs=5)

# Test the model
print("\nTesting the model...")
test_accuracy = test_model()

# Plot training progress
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, 'b-', label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, 'r-', label='Training Accuracy')
plt.title('Training Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Test on some individual images
def test_individual_predictions():
    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Show first 8 test images with predictions
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(8):
        row = i // 4
        col = i % 4
        axes[row, col].imshow(images[i].squeeze(), cmap='gray')
        axes[row, col].set_title(f'True: {labels[i].item()}, Pred: {predicted[i].item()}')
        axes[row, col].axis('off')
        
        # Color the title based on correctness
        if labels[i].item() == predicted[i].item():
            axes[row, col].title.set_color('green')
        else:
            axes[row, col].title.set_color('red')
    
    plt.suptitle('Model Predictions on Test Images')
    plt.tight_layout()
    plt.show()
    
    

test_individual_predictions()

