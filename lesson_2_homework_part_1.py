# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 23:29:48 2025

@author: Gavin

Instructions:
- Complete the TODO sections marked below
- Run experiments with different hidden layer sizes
- Analyze the results and answer questions in homework_questions.md
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

# Define transforms for preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Download and load MNIST dataset
print("Loading MNIST dataset...")
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

# Modified neural network class with variable hidden layer size
class VariableMNISTNet(nn.Module):
    
    def __init__(self, hidden_size=128):
        """
        Initialize the network with variable hidden layer size
        
        Args:
            hidden_size (int): Number of neurons in the first hidden layer
        """
        super(VariableMNISTNet, self).__init__()
        self.flatten = nn.Flatten()
        
        # Network layers with variable first hidden layer
        self.fc1 = nn.Linear(28 * 28, hidden_size)  # Variable first hidden layer
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 64)       # Second hidden layer
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)                # Output layer
        
        # Store hidden size for reference
        self.hidden_size = hidden_size
    
    
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    
    

def train_model(model, num_epochs=3):
    """
    Train the model and return final test accuracy
    
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
            
            # Print progress every 200 batches (less frequent for homework)
            if batch_idx % 200 == 0:
                print(f'Hidden Size: {model.hidden_size}, Epoch: {epoch+1}/{num_epochs}, '
                      f'Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        epoch_acc = correct / total
        print(f'Hidden Size: {model.hidden_size}, Epoch {epoch+1}/{num_epochs}, '
              f'Training Accuracy: {epoch_acc:.4f}')
    
    # Test the model
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    test_accuracy = correct / total
    print(f'Hidden Size: {model.hidden_size}, Test Accuracy: {test_accuracy:.4f}\n')
    
    return test_accuracy



# TODO 1: Complete the experiment loop
# Instructions:
# 1. Define a list of hidden layer sizes to test (suggested: [1, 2, 4, 8, 16, 32, 64, 128])
# 2. Create a loop that trains a model for each hidden size
# 3. Store the results in the provided lists
# 4. Print progress as you go

print("Starting architecture investigation...")

# TODO: Define your hidden layer sizes to test here
hidden_sizes = []  # Replace with your list of sizes, e.g., [1, 2, 4, 8, 16, 32, 64, 128]

# Lists to store results
results_hidden_sizes = []
results_accuracies = []

# TODO: Write your experiment loop here
# PSEUDOCODE:
# for each_hidden_size in hidden_sizes:
#     print(f"Testing hidden size: {each_hidden_size}")
#     model = VariableMNISTNet(hidden_size=each_hidden_size)
#     accuracy = train_model(model, num_epochs=3)
#     results_hidden_sizes.append(each_hidden_size)
#     results_accuracies.append(accuracy)

# Your code here:


# TODO 2: Complete the plotting function
def plot_architecture_results(hidden_sizes, accuracies):
    """
    Plot how test accuracy varies with hidden layer size
    
    Args:
        hidden_sizes (list): List of hidden layer sizes tested
        accuracies (list): List of corresponding test accuracies
    """
    plt.figure(figsize=(10, 6))
    
    # Create both line and scatter plot for better visualization
    plt.plot(hidden_sizes, accuracies, 'b-o', linewidth=2, markersize=8, label='Test Accuracy')
    
    # Add labels and title
    plt.xlabel('Hidden Layer Size', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('Effect of Hidden Layer Size on MNIST Classification Accuracy', fontsize=14)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Format the plot
    plt.legend()
    plt.tight_layout()
    
    # Add some styling - adjust y-axis for wider range of performance
    plt.ylim(0.1, 1.0)  # Allow for very poor performance with tiny networks
    
    # Use log scale for x-axis to better show the range from 1 to 128
    plt.xscale('log')
    plt.xticks(hidden_sizes, hidden_sizes)  # Show all tested sizes
    
    # Add value annotations on points
    for i, (size, acc) in enumerate(zip(hidden_sizes, accuracies)):
        plt.annotate(f'{acc:.3f}', (size, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    plt.show()
    
    # Print summary statistics
    best_idx = np.argmax(accuracies)
    print(f"\nSUMMARY:")
    print(f"Best hidden layer size: {hidden_sizes[best_idx]} neurons")
    print(f"Best accuracy: {accuracies[best_idx]:.4f}")
    print(f"Accuracy range: {min(accuracies):.4f} - {max(accuracies):.4f}")



# TODO 3: Call your plotting function with the results
# Uncomment the line below once you've completed TODO 1
# plot_architecture_results(results_hidden_sizes, results_accuracies)

# TODO 4: Analysis
# After running your experiments, add your observations here as comments:
# 1. Which hidden layer size performed best?
# 2. What happens with very small hidden layers (e.g., 16)?
# 3. What happens with very large hidden layers (e.g., 512)?
# 4. Is there a pattern you can observe?

print("Homework Part 1 Complete!")
print("Don't forget to answer the questions in lesson_2_homework_part_3.md")