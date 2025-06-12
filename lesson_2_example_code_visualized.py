# -*- coding: utf-8 -*-
"""
Enhanced MNIST Neural Network Demo with Visual Intuition
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
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

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

# Enhanced function to show sample images with data distribution
def show_sample_images_and_distribution():
    # Get a batch of training data
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    
    # Create subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Plot sample images
    for i in range(12):
        plt.subplot(3, 6, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f'Label: {labels[i].item()}')
        plt.axis('off')
    
    plt.suptitle('Sample MNIST Images', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Show class distribution
    all_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    plt.figure(figsize=(10, 6))
    plt.hist(all_labels, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Digit Class')
    plt.ylabel('Frequency')
    plt.title('Distribution of Digits in Training Dataset')
    plt.xticks(range(10))
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Show pixel intensity distribution for different digits
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for digit in range(10):
        row = digit // 5
        col = digit % 5
        
        # Get all images of this digit
        digit_images = []
        for img, label in train_dataset:
            if label == digit:
                digit_images.append(img.numpy().flatten())
                if len(digit_images) >= 100:  # Sample 100 images per digit
                    break
        
        if digit_images:
            pixel_intensities = np.concatenate(digit_images)
            axes[row, col].hist(pixel_intensities, bins=30, alpha=0.7, density=True)
            axes[row, col].set_title(f'Digit {digit}')
            axes[row, col].set_xlabel('Pixel Intensity')
            axes[row, col].set_ylabel('Density')
    
    plt.suptitle('Pixel Intensity Distributions by Digit', fontsize=16)
    plt.tight_layout()
    plt.show()

show_sample_images_and_distribution()

# Define a simple neural network with visualization capabilities
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
    
    def forward_with_activations(self, x):
        """Forward pass that returns intermediate activations for visualization"""
        activations = {}
        x = self.flatten(x)
        activations['input'] = x.clone()
        
        x = self.fc1(x)
        activations['fc1_pre'] = x.clone()
        x = self.relu1(x)
        activations['fc1_post'] = x.clone()
        
        x = self.fc2(x)
        activations['fc2_pre'] = x.clone()
        x = self.relu2(x)
        activations['fc2_post'] = x.clone()
        
        x = self.fc3(x)
        activations['output'] = x.clone()
        
        return x, activations

# Create the model
model = MNISTNet()
print("Model Architecture:")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to visualize network weights
def visualize_weights():
    # Visualize first layer weights as images
    weights = model.fc1.weight.data.cpu().numpy()  # Shape: (128, 784)
    
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for i in range(32):  # Show first 32 neurons
        row = i // 8
        col = i % 8
        weight_img = weights[i].reshape(28, 28)
        im = axes[row, col].imshow(weight_img, cmap='RdBu', vmin=-weight_img.std(), vmax=weight_img.std())
        axes[row, col].set_title(f'Neuron {i}')
        axes[row, col].axis('off')
    
    plt.suptitle('First Layer Weights (What Each Neuron "Looks For")', fontsize=16)
    plt.tight_layout()
    plt.show()

# Function to visualize activations
def visualize_activations(sample_image, sample_label):
    model.eval()
    with torch.no_grad():
        output, activations = model.forward_with_activations(sample_image.unsqueeze(0))
        
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(sample_image.squeeze(), cmap='gray')
    axes[0, 0].set_title(f'Input Image (Label: {sample_label})')
    axes[0, 0].axis('off')
    
    # First layer activations (before ReLU)
    fc1_pre = activations['fc1_pre'].squeeze().numpy()
    axes[0, 1].bar(range(len(fc1_pre[:20])), fc1_pre[:20])
    axes[0, 1].set_title('First Layer (Before ReLU) - First 20 neurons')
    axes[0, 1].set_xlabel('Neuron Index')
    axes[0, 1].set_ylabel('Activation')
    
    # First layer activations (after ReLU)
    fc1_post = activations['fc1_post'].squeeze().numpy()
    axes[0, 2].bar(range(len(fc1_post[:20])), fc1_post[:20])
    axes[0, 2].set_title('First Layer (After ReLU) - First 20 neurons')
    axes[0, 2].set_xlabel('Neuron Index')
    axes[0, 2].set_ylabel('Activation')
    
    # Second layer activations
    fc2_post = activations['fc2_post'].squeeze().numpy()
    axes[1, 0].bar(range(len(fc2_post)), fc2_post)
    axes[1, 0].set_title('Second Layer Activations')
    axes[1, 0].set_xlabel('Neuron Index')
    axes[1, 0].set_ylabel('Activation')
    
    # Output layer (raw scores)
    output_vals = activations['output'].squeeze().numpy()
    axes[1, 1].bar(range(10), output_vals)
    axes[1, 1].set_title('Output Layer (Raw Scores)')
    axes[1, 1].set_xlabel('Digit Class')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_xticks(range(10))
    
    # Output probabilities
    probabilities = torch.softmax(torch.tensor(output_vals), dim=0).numpy()
    bars = axes[1, 2].bar(range(10), probabilities)
    axes[1, 2].set_title('Output Probabilities')
    axes[1, 2].set_xlabel('Digit Class')
    axes[1, 2].set_ylabel('Probability')
    axes[1, 2].set_xticks(range(10))
    
    # Highlight the predicted class
    predicted_class = np.argmax(probabilities)
    bars[predicted_class].set_color('red')
    bars[sample_label].set_color('green' if predicted_class == sample_label else 'orange')
    
    plt.tight_layout()
    plt.show()
    
    return predicted_class

# Enhanced training function with live visualization
def train_model_with_visualization(num_epochs=5):
    model.train()
    train_losses = []
    train_accuracies = []
    
    # Get a sample for visualization
    sample_data = next(iter(test_loader))
    sample_image, sample_label = sample_data[0][0], sample_data[1][0]
    
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
            
            # Print progress every 200 batches
            if batch_idx % 200 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}, '
                      f'Loss: {loss.item():.4f}')
        
        # Calculate epoch statistics
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Average Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
        
        # Visualize progress every epoch
        if epoch == 0:
            print("Visualizing initial weights...")
            visualize_weights()
            print("Visualizing initial predictions...")
            visualize_activations(sample_image, sample_label.item())
    
    # Final visualizations
    print("Visualizing final weights...")
    visualize_weights()
    print("Visualizing final predictions...")
    visualize_activations(sample_image, sample_label.item())
    
    return train_losses, train_accuracies

# Test function
def test_model():
    model.eval()
    correct = 0
    total = 0
    all_predicted = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            all_predicted.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f} ({100 * accuracy:.2f}%)')
    
    return accuracy, all_predicted, all_targets

# Function to visualize confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    # Calculate per-class accuracy
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(10), class_accuracies)
    plt.title('Per-Class Accuracy')
    plt.xlabel('Digit Class')
    plt.ylabel('Accuracy')
    plt.xticks(range(10))
    plt.ylim(0, 1)
    
    # Color bars based on performance
    for i, (bar, acc) in enumerate(zip(bars, class_accuracies)):
        if acc > 0.95:
            bar.set_color('green')
        elif acc > 0.90:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.grid(True, alpha=0.3)
    plt.show()

# Train the model
print("Starting training...")
train_losses, train_accuracies = train_model_with_visualization(num_epochs=5)

# Test the model
print("\nTesting the model...")
test_accuracy, predicted_labels, true_labels = test_model()

# Plot training progress
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_losses, 'b-', label='Training Loss', linewidth=2)
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(train_accuracies, 'r-', label='Training Accuracy', linewidth=2)
plt.title('Training Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
epochs = range(1, len(train_losses) + 1)
plt.plot(epochs, train_losses, 'b-', label='Loss', linewidth=2)
plt.plot(epochs, train_accuracies, 'r-', label='Accuracy', linewidth=2)
plt.title('Training Progress')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Visualize confusion matrix
plot_confusion_matrix(true_labels, predicted_labels)

# Test on individual images with detailed analysis
def test_individual_predictions_detailed():
    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    # Show predictions with confidence scores
    fig, axes = plt.subplots(3, 6, figsize=(18, 12))
    for i in range(18):
        row = i // 6
        col = i % 6
        
        axes[row, col].imshow(images[i].squeeze(), cmap='gray')
        
        confidence = probabilities[i][predicted[i]].item()
        true_label = labels[i].item()
        pred_label = predicted[i].item()
        
        title = f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.3f}'
        axes[row, col].set_title(title, fontsize=10)
        axes[row, col].axis('off')
        
        # Color the title based on correctness and confidence
        if true_label == pred_label:
            if confidence > 0.9:
                axes[row, col].title.set_color('darkgreen')
            else:
                axes[row, col].title.set_color('green')
        else:
            if confidence > 0.5:
                axes[row, col].title.set_color('red')
            else:
                axes[row, col].title.set_color('orange')
    
    plt.suptitle('Model Predictions with Confidence Scores\n'
                 'Dark Green: Correct & High Confidence, Green: Correct & Low Confidence\n'
                 'Red: Wrong & High Confidence, Orange: Wrong & Low Confidence', 
                 fontsize=14)
    plt.tight_layout()
    plt.show()

test_individual_predictions_detailed()

# Analyze misclassified examples
def analyze_misclassifications():
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_predictions = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            # Find misclassified examples
            mask = predicted != targets
            if mask.any():
                misclassified_images.extend(data[mask])
                misclassified_labels.extend(targets[mask])
                misclassified_predictions.extend(predicted[mask])
                
                if len(misclassified_images) >= 12:  # Get first 12 misclassifications
                    break
    
    if misclassified_images:
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        for i in range(min(12, len(misclassified_images))):
            row = i // 4
            col = i % 4
            
            axes[row, col].imshow(misclassified_images[i].squeeze(), cmap='gray')
            axes[row, col].set_title(f'True: {misclassified_labels[i].item()}, '
                                   f'Pred: {misclassified_predictions[i].item()}')
            axes[row, col].axis('off')
        
        plt.suptitle('Examples of Misclassified Images', fontsize=16)
        plt.tight_layout()
        plt.show()

analyze_misclassifications()

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Final Test Accuracy: {test_accuracy:.4f} ({100 * test_accuracy:.2f}%)")
print("Check the visualizations above to understand:")
print("1. How the network weights evolved during training")
print("2. How different layers activate for different inputs")
print("3. Which digits the model struggles with most")
print("4. Examples of confident vs uncertain predictions")
print("="*60)