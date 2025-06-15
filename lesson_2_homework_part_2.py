# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 23:29:48 2025

@author: Gavin

Instructions:
- Complete the TODO sections to adapt the network for color images
- CIFAR-10 contains 32x32 color images (3 channels: RGB)
- Compare performance with the MNIST results
- Answer questions in homework_questions.md
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# CIFAR-10 class names for reference
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Define transforms for preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    # Note: CIFAR-10 images are already small (32x32), so no resizing needed
])

# Download and load CIFAR-10 dataset
print("Loading CIFAR-10 dataset...")
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Function to show sample CIFAR-10 images
def show_sample_images():
    """Display sample CIFAR-10 images with their labels"""
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    
    # Plot first 8 images
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(8):
        row = i // 4
        col = i % 4
        
        # Convert tensor to numpy and rearrange dimensions for display
        img = images[i].permute(1, 2, 0)  # Change from (C,H,W) to (H,W,C)
        axes[row, col].imshow(img)
        axes[row, col].set_title(f'Label: {class_names[labels[i].item()]}')
        axes[row, col].axis('off')
    
    plt.suptitle('Sample CIFAR-10 Images')
    plt.tight_layout()
    plt.show()
    
    

show_sample_images()

# TODO 1: Design the network for color images
class CIFAR10Net(nn.Module):
    
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        
        # TODO: Think about the input size for color images
        # MNIST: 28x28x1 (grayscale) = 784 input features
        # CIFAR-10: 32x32x3 (RGB) = ? input features
        
        self.flatten = nn.Flatten()
        
        # TODO: Complete the network architecture
        # Hints:
        # 1. Calculate the correct input size for the first layer
        # 2. CIFAR-10 has 10 classes (same as MNIST)
        # 3. You might want to use more neurons since color images are more complex
        # 4. Consider the network depth - you might need more layers
        
        # Your network layers here:
        # self.fc1 = nn.Linear(???, ???)  # First layer - what should the input size be?
        # Add more layers as needed...
        
        pass  # Remove this once you add your layers
    
    
    
    def forward(self, x):
        # TODO: Complete the forward pass
        # Make sure to apply the layers in the correct order
        
        # Your forward pass here:
        pass  # Remove this once you implement the forward pass
    
    

# TODO 2: Training function
def train_cifar_model(model, num_epochs=5):
    """
    Train the CIFAR-10 model
    
    Args:
        model: The neural network model to train
        num_epochs (int): Number of epochs to train for
        
    Returns:
        float: Final test accuracy
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            # TODO: Complete the training loop
            # PSEUDOCODE:
            # outputs = model(data)                    # Forward pass
            # loss = criterion(outputs, targets)       # Calculate loss
            # optimizer.zero_grad()                    # Clear gradients
            # loss.backward()                          # Backward pass
            # optimizer.step()                         # Update weights
            # 
            # # Statistics:
            # total_loss += loss.item()
            # _, predicted = torch.max(outputs.data, 1)
            # total += targets.size(0)
            # correct += (predicted == targets).sum().item()
            
            # Your training code here:
            pass  # Remove this once you implement the training loop
            
            # Print progress every 200 batches
            if batch_idx % 200 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}, '
                      f'Loss: {loss.item():.4f}')
        
        # Calculate and print epoch statistics
        epoch_acc = correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Training Accuracy: {epoch_acc:.4f}')
    
    # TODO: Implement testing
    # After training, evaluate the model on the test set
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            # TODO: Complete the testing loop
            # Your testing code here:
            pass  # Remove this once you implement testing
    
    test_accuracy = test_correct / test_total
    print(f'Final Test Accuracy: {test_accuracy:.4f} ({100 * test_accuracy:.2f}%)')
    
    return test_accuracy



# TODO 3: Create and train your model
# Uncomment and complete the following lines once you've implemented the above

# print("Creating CIFAR-10 model...")
# model = CIFAR10Net()
# print("Model Architecture:")
# print(model)

# print("\nStarting training...")
# final_accuracy = train_cifar_model(model, num_epochs=5)

# print("\nAnalyzing results...")
# show_predictions(model)
# analyze_class_performance(model)

# TODO 4: Visualization function for predictions
def show_predictions(model):
    """Show model predictions on test images"""
    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Show first 8 test images with predictions
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    for i in range(8):
        row = i // 4
        col = i % 4
        
        # Convert tensor to numpy and rearrange dimensions for display
        img = images[i].permute(1, 2, 0)  # Change from (C,H,W) to (H,W,C)
        axes[row, col].imshow(img)
        
        true_label = class_names[labels[i].item()]
        pred_label = class_names[predicted[i].item()]
        
        axes[row, col].set_title(f'True: {true_label}\nPred: {pred_label}', fontsize=10)
        axes[row, col].axis('off')
        
        # Color the title based on correctness
        if labels[i].item() == predicted[i].item():
            axes[row, col].title.set_color('green')
        else:
            axes[row, col].title.set_color('red')
    
    plt.suptitle('CIFAR-10 Model Predictions on Test Images', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    

# Function to analyze model performance by class
def analyze_class_performance(model):
    """Analyze model performance for each CIFAR-10 class"""
    model.eval()
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == target).squeeze()
            for i in range(target.size(0)):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    print("\nPer-class accuracy:")
    for i in range(10):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f'{class_names[i]}: {accuracy:.1f}% ({int(class_correct[i])}/{int(class_total[i])})')
    
    # Plot class accuracies
    class_accuracies = [100 * class_correct[i] / class_total[i] for i in range(10)]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, class_accuracies, color='skyblue', edgecolor='navy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title('CIFAR-10 Classification Accuracy by Class')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, class_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()



# TODO 5: Analysis Questions (answer in homework_questions.md)
# Consider these questions as you work:
# 1. How does the input size change from MNIST to CIFAR-10?
# 2. How does the accuracy compare between MNIST and CIFAR-10?
# 3. Why might CIFAR-10 be more challenging than MNIST?
# 4. What network architecture choices did you make and why?

print("Homework Part 2 Template Ready!")
print("Complete the TODOs above, then run your experiments.")
print("Don't forget to answer the questions in lesson_2_homework_part_3.md")