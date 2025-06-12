# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 19:25:18 2025
@author: Gavin

Basic CNN for MNIST - Beginner Version with Visual Examples
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

# Let's first look at our data!
def show_sample_images():
    """Show some example MNIST images to understand what we're working with"""
    print("Let's look at some MNIST images first!")
    
    # Get a batch of images
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    
    # Show 8 images
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    fig.suptitle('Sample MNIST Images')
    
    for i in range(8):
        row = i // 4
        col = i % 4
        
        # Convert from tensor to numpy for plotting
        img = images[i].squeeze()  # Remove the channel dimension
        
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(f'Label: {labels[i].item()}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Image shape: {images[0].shape}")  # Should be [1, 28, 28]
    print(f"Batch shape: {images.shape}")     # Should be [64, 1, 28, 28]



# Define a SIMPLE Convolutional Neural Network
class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        
        # Layer 1: Convolution
        # Input: 1 channel (grayscale), Output: 8 channels
        # Kernel size: 5x5, Padding: 2 (keeps size same)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=2)
        
        # Layer 2: Max pooling (reduces size by half)
        self.pool = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        
        # Layer 3: Another convolution
        # Input: 8 channels, Output: 16 channels
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        # After this + pooling: 14x14 -> 7x7
        
        # Layer 4: Fully connected layers
        # After conv layers: 16 channels * 7 * 7 = 784 features
        self.fc1 = nn.Linear(16 * 7 * 7, 32)  # Hidden layer with 32 neurons
        self.fc2 = nn.Linear(32, 10)           # Output layer (10 digits)
        
        # Activation function
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Show what happens at each step!
        
        # Step 1: First convolution + activation
        x = self.relu(self.conv1(x))  # [64, 1, 28, 28] -> [64, 8, 28, 28]
        
        # Step 2: First pooling
        x = self.pool(x)              # [64, 8, 28, 28] -> [64, 8, 14, 14]
        
        # Step 3: Second convolution + activation
        x = self.relu(self.conv2(x))  # [64, 8, 14, 14] -> [64, 16, 14, 14]
        
        # Step 4: Second pooling
        x = self.pool(x)              # [64, 16, 14, 14] -> [64, 16, 7, 7]
        
        # Step 5: Flatten for fully connected layers
        x = x.view(-1, 16 * 7 * 7)    # [64, 16, 7, 7] -> [64, 784]
        
        # Step 6: First fully connected layer
        x = self.relu(self.fc1(x))    # [64, 784] -> [64, 32]
        
        # Step 7: Final output layer
        x = self.fc2(x)               # [64, 32] -> [64, 10]
        
        return x
    
    

# Create model, loss function, and optimizer
model = MNISTCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Let's visualize the network architecture
def show_model_info():
    """Show information about our model"""
    print("=== MODEL ARCHITECTURE ===")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Show what happens to image size at each layer
    print("\n=== WHAT HAPPENS TO IMAGE SIZE ===")
    print("Input image:        [1, 28, 28]")
    print("After conv1:        [8, 28, 28]   (8 feature maps)")
    print("After pool1:        [8, 14, 14]   (size halved)")
    print("After conv2:        [16, 14, 14]  (16 feature maps)")
    print("After pool2:        [16, 7, 7]    (size halved again)")
    print("After flatten:      [784]         (16*7*7 = 784)")
    print("After fc1:          [32]          (hidden layer)")
    print("After fc2:          [10]          (final output - 10 digits)")



# Training function with visual feedback
def train_cnn(num_epochs=3):
    """Train the CNN and show progress"""
    print(f"\n=== TRAINING FOR {num_epochs} EPOCHS ===")
    
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Show progress every 200 batches
            if (i + 1) % 200 == 0:
                print(f'Batch {i+1:3d}: Loss = {loss.item():.4f}, Accuracy = {100*correct/total:.2f}%')
        
        # Epoch summary
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        losses.append(epoch_loss)
        
        print(f'Epoch {epoch+1} Summary: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.2f}%')
    
    # Plot training loss
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, num_epochs+1), losses, 'b-', marker='o')
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    return losses



# Testing function
def test_cnn():
    """Test the CNN and show results"""
    print("\n=== TESTING THE MODEL ===")
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # Show some test predictions
    show_test_predictions()
    
    return accuracy



def show_test_predictions():
    """Show some test predictions to see how well the model works"""
    model.eval()
    
    # Get a batch of test images
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Show 8 examples
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle('Test Predictions vs Actual Labels')
    
    for i in range(8):
        row = i // 4
        col = i % 4
        
        # Convert image to numpy
        img = images[i].squeeze()
        
        # Check if prediction is correct
        is_correct = predicted[i] == labels[i]
        color = 'green' if is_correct else 'red'
        
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(f'Pred: {predicted[i].item()}, True: {labels[i].item()}', color=color)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    

# Simple visualization of what the filters learned
def visualize_conv_filters():
    """Show what the first layer filters look like"""
    print("\n=== VISUALIZING LEARNED FILTERS ===")
    
    # Get the weights from first convolutional layer
    conv1_weights = model.conv1.weight.data  # Shape: [8, 1, 5, 5]
    
    # Plot all 8 filters
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    fig.suptitle('Learned Convolutional Filters (First Layer)')
    
    for i in range(8):
        row = i // 4
        col = i % 4
        
        # Get the filter (remove the input channel dimension)
        filter_img = conv1_weights[i, 0, :, :].cpu().numpy()
        
        axes[row, col].imshow(filter_img, cmap='gray')
        axes[row, col].set_title(f'Filter {i+1}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("These filters detect different patterns like edges, corners, etc.")



# MAIN EXECUTION
if __name__ == "__main__":
    print("MNIST CNN Training - Beginner Tutorial")
    print("=" * 40)
    
    # Step 1: Look at the data
    show_sample_images()
    
    # Step 2: Understand the model
    show_model_info()
    
    # Step 3: Train the model
    print("\nReady to train? This will take a few minutes...")
    input("Press Enter to start training...")
    
    losses = train_cnn(num_epochs=3)
    
    # Step 4: Test the model
    accuracy = test_cnn()
    
    # Step 5: See what the model learned
    visualize_conv_filters()
    
    print(f"\n🎉 Training complete! Final accuracy: {accuracy:.2f}%")
    print("The CNN learned to recognize handwritten digits!")
    
    