# -*- coding: utf-8 -*-
"""
MNIST CNN Exercise Solutions - Multiple Approaches
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
import numpy as np

# Data loading and preprocessing - this is important!
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL Image (0-255) to PyTorch tensor (0.0-1.0)
    # Note: ToTensor() automatically normalizes to 0-1 range, so we're good to go!
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# =============================================================================
# SOLUTION 1: BASIC CNN (Good for beginners)
# =============================================================================
class MNISTCNN_Basic(nn.Module):
    def __init__(self):
        super(MNISTCNN_Basic, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.pool1 = nn.MaxPool2d(2, 2)                          # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 14x14 -> 14x14
        self.pool2 = nn.MaxPool2d(2, 2)                          # 14x14 -> 7x7
        
        # Fully connected layers
        # After pooling: 64 channels * 7 * 7 = 3136 features
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 64*7*7)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# =============================================================================
# SOLUTION 2: INTERMEDIATE CNN (More sophisticated)
# =============================================================================
class MNISTCNN_Intermediate(nn.Module):
    def __init__(self):
        super(MNISTCNN_Intermediate, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 28x28 -> 28x28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                           # 28x28 -> 14x14
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 14x14 -> 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                           # 14x14 -> 7x7
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 7x7 -> 7x7
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))                  # 7x7 -> 4x4
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# =============================================================================
# SOLUTION 3: ADVANCED CNN (ResNet-inspired)
# =============================================================================
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class MNISTCNN_Advanced(nn.Module):
    def __init__(self):
        super(MNISTCNN_Advanced, self).__init__()
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

# =============================================================================
# TRAINING AND TESTING FUNCTIONS
# =============================================================================

def train_cnn(model, num_epochs=5, model_name="CNN"):
    """Enhanced training function with detailed tracking"""
    model.train()
    train_losses = []
    train_accuracies = []
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training {model_name}...")
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
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
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Progress update
            if batch_idx % 200 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}, '
                      f'Loss: {loss.item():.4f}')
        
        # Epoch statistics
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        print(f'{model_name} - Epoch {epoch+1}/{num_epochs}: '
              f'Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    
    return train_losses, train_accuracies

def test_cnn(model, model_name="CNN"):
    """Test the CNN model"""
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Per-class accuracy
            c = (predicted == targets).squeeze()
            for i in range(targets.size(0)):
                label = targets[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    overall_accuracy = correct / total
    print(f'\n{model_name} Test Results:')
    print(f'Overall Accuracy: {overall_accuracy:.4f} ({100 * overall_accuracy:.2f}%)')
    
    # Per-class accuracy
    print('\nPer-class Accuracy:')
    for i in range(10):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            print(f'Digit {i}: {acc:.4f} ({100 * acc:.1f}%)')
    
    return overall_accuracy

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def visualize_conv_filters(model, model_name="CNN"):
    """Visualize the learned convolutional filters"""
    print(f"\nVisualizing {model_name} filters...")
    
    # Get the first convolutional layer
    if hasattr(model, 'conv1'):
        first_conv = model.conv1
    elif hasattr(model, 'features') and len(model.features) > 0:
        first_conv = model.features[0]
    else:
        print("Could not find first convolutional layer")
        return
    
    # Extract filters
    filters = first_conv.weight.data.cpu().numpy()
    
    # Plot filters
    num_filters = min(filters.shape[0], 32)  # Show up to 32 filters
    cols = 8
    rows = (num_filters + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 2*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_filters):
        row = i // cols
        col = i % cols
        filter_img = filters[i, 0]  # First channel (grayscale)
        
        im = axes[row, col].imshow(filter_img, cmap='RdBu', 
                                  vmin=-np.abs(filter_img).max(), 
                                  vmax=np.abs(filter_img).max())
        axes[row, col].set_title(f'Filter {i}')
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(num_filters, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'{model_name} - First Layer Convolutional Filters', fontsize=16)
    plt.tight_layout()
    plt.show()

def visualize_feature_maps(model, model_name="CNN"):
    """Visualize feature maps for a sample image"""
    model.eval()
    
    # Get a sample image
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    sample_image = images[0:1]  # Take first image, keep batch dimension
    sample_label = labels[0].item()
    
    # Forward pass to get intermediate activations
    activations = []
    x = sample_image
    
    with torch.no_grad():
        if hasattr(model, 'features'):
            # For models with features module
            for i, layer in enumerate(model.features):
                x = layer(x)
                if isinstance(layer, nn.Conv2d):
                    activations.append(x.clone())
        else:
            # For basic model
            if hasattr(model, 'conv1'):
                x = model.relu(model.conv1(x))
                activations.append(x.clone())
                x = model.pool1(x)
                
                x = model.relu(model.conv2(x))
                activations.append(x.clone())
    
    # Plot original image and feature maps
    fig, axes = plt.subplots(len(activations) + 1, 8, figsize=(16, 2*(len(activations) + 1)))
    
    # Original image
    axes[0, 0].imshow(sample_image.squeeze(), cmap='gray')
    axes[0, 0].set_title(f'Original (Label: {sample_label})')
    axes[0, 0].axis('off')
    
    # Hide remaining slots in first row
    for j in range(1, 8):
        axes[0, j].axis('off')
    
    # Feature maps
    for i, activation in enumerate(activations):
        feature_maps = activation.squeeze().cpu().numpy()
        for j in range(min(8, feature_maps.shape[0])):
            axes[i+1, j].imshow(feature_maps[j], cmap='viridis')
            axes[i+1, j].set_title(f'Layer {i+1}, Filter {j}')
            axes[i+1, j].axis('off')
        
        # Hide remaining slots
        for j in range(min(8, feature_maps.shape[0]), 8):
            axes[i+1, j].axis('off')
    
    plt.suptitle(f'{model_name} - Feature Maps Visualization', fontsize=16)
    plt.tight_layout()
    plt.show()

def compare_models(models_dict):
    """Compare multiple models performance"""
    results = {}
    
    for name, model in models_dict.items():
        print(f"\n{'='*50}")
        print(f"TRAINING AND TESTING: {name}")
        print(f"{'='*50}")
        
        # Train model
        train_losses, train_accuracies = train_cnn(model, num_epochs=5, model_name=name)
        
        # Test model
        test_accuracy = test_cnn(model, model_name=name)
        
        # Store results
        results[name] = {
            'model': model,
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_accuracy': test_accuracy
        }
        
        # Visualizations
        visualize_conv_filters(model, model_name=name)
        visualize_feature_maps(model, model_name=name)
    
    # Plot comparison
    plt.figure(figsize=(15, 5))
    
    # Training losses
    plt.subplot(1, 3, 1)
    for name, result in results.items():
        plt.plot(result['train_losses'], label=name, linewidth=2)
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Training accuracies
    plt.subplot(1, 3, 2)
    for name, result in results.items():
        plt.plot(result['train_accuracies'], label=name, linewidth=2)
    plt.title('Training Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Test accuracies
    plt.subplot(1, 3, 3)
    names = list(results.keys())
    test_accs = [results[name]['test_accuracy'] for name in names]
    bars = plt.bar(names, test_accs, color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.title('Test Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars, test_accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return results

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("MNIST CNN Solutions - Multiple Approaches")
    print("=" * 60)
    
    # Create different models
    models = {
        'Basic CNN': MNISTCNN_Basic(),
        'Intermediate CNN': MNISTCNN_Intermediate(),
        'Advanced CNN': MNISTCNN_Advanced()
    }
    
    # Option 1: Run all models for comparison
    print("Running comprehensive comparison of all models...")
    results = compare_models(models)
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    for name, result in results.items():
        print(f"{name}: {result['test_accuracy']:.4f} ({100 * result['test_accuracy']:.2f}%)")
    
    # Option 2: If you want to run just one model, uncomment below:
    '''
    print("Running Basic CNN only...")
    model = MNISTCNN_Basic()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train and test
    train_losses, train_accuracies = train_cnn(model, num_epochs=5, model_name="Basic CNN")
    test_accuracy = test_cnn(model, model_name="Basic CNN")
    
    # Visualize
    visualize_conv_filters(model, model_name="Basic CNN")
    visualize_feature_maps(model, model_name="Basic CNN")
    '''

# =============================================================================
# ADDITIONAL CHALLENGE SOLUTIONS
# =============================================================================

def advanced_filter_analysis(model, model_name="CNN"):
    """Advanced analysis of what filters learn"""
    print(f"\nAdvanced Filter Analysis for {model_name}")
    
    # Get first conv layer
    if hasattr(model, 'conv1'):
        first_conv = model.conv1
    elif hasattr(model, 'features'):
        first_conv = model.features[0]
    else:
        return
    
    filters = first_conv.weight.data.cpu().numpy()
    
    # Analyze filter statistics
    print(f"Filter shape: {filters.shape}")
    print(f"Mean filter value: {filters.mean():.6f}")
    print(f"Std filter value: {filters.std():.6f}")
    print(f"Min filter value: {filters.min():.6f}")
    print(f"Max filter value: {filters.max():.6f}")
    
    # Plot filter statistics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Histogram of all filter values
    axes[0, 0].hist(filters.flatten(), bins=50, alpha=0.7)
    axes[0, 0].set_title('Distribution of All Filter Values')
    axes[0, 0].set_xlabel('Weight Value')
    axes[0, 0].set_ylabel('Frequency')
    
    # Filter norms
    filter_norms = np.linalg.norm(filters.reshape(filters.shape[0], -1), axis=1)
    axes[0, 1].bar(range(len(filter_norms)), filter_norms)
    axes[0, 1].set_title('L2 Norm of Each Filter')
    axes[0, 1].set_xlabel('Filter Index')
    axes[0, 1].set_ylabel('L2 Norm')
    
    # Mean and std per filter
    filter_means = filters.mean(axis=(1, 2, 3))
    filter_stds = filters.std(axis=(1, 2, 3))
    
    axes[1, 0].scatter(filter_means, filter_stds)
    axes[1, 0].set_title('Filter Mean vs Std')
    axes[1, 0].set_xlabel('Mean')
    axes[1, 0].set_ylabel('Standard Deviation')
    
    # Filter similarity matrix
    n_filters = min(filters.shape[0], 16)
    similarity_matrix = np.zeros((n_filters, n_filters))
    for i in range(n_filters):
        for j in range(n_filters):
            f1 = filters[i].flatten()
            f2 = filters[j].flatten()
            similarity_matrix[i, j] = np.corrcoef(f1, f2)[0, 1]
    
    im = axes[1, 1].imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 1].set_title('Filter Similarity Matrix')
    axes[1, 1].set_xlabel('Filter Index')
    axes[1, 1].set_ylabel('Filter Index')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()

# Uncomment to run advanced analysis:
# advanced_filter_analysis(models['Basic CNN'], "Basic CNN")