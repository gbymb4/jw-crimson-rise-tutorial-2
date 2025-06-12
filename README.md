# Deep Learning with PyTorch - Session 2

## Session Timeline

| Time      | Activity                                |
| --------- | --------------------------------------- |
| 0:00 - 0:10 | 1. Check-in + Session 1 Recap          |
| 0:10 - 0:20 | 2. Run the Full MNIST Example Together |
| 0:20 - 0:45 | 3. Explain the Code (Discussion/Slides) |
| 0:45 - 1:15 | 4. Guided Hands-On Practice             |
| 1:15 - 1:50 | 5. Independent Challenge                |
| 1:50 - 2:00 | 6. Wrap-Up + Homework                   |

---

## 1. Check-in + Session 1 Recap

### Goals

Let's reconnect and review what we learned in Session 1 before diving into image classification with neural networks!

We'll quickly revisit:

* Linear regression with PyTorch - how we learned `y = 2x + 10`
* Key concepts: tensors, parameters, loss functions, optimizers
* The training loop: forward pass, loss calculation, backpropagation, parameter updates
* How your homework projects went (multiple features, batch training, etc.)

### Quick Recap Questions

* What did `torch.nn.Parameter` do in our linear model?
* Can you explain the training loop steps in your own words?
* What was the difference between the loss function and the optimizer?
* How did changing the learning rate affect training?

### What's New Today

Today we're moving from simple linear regression to **image classification**! We'll use the famous MNIST dataset (handwritten digits 0-9) to build a neural network that can recognize what digit is in an image.

This is a big step up - instead of learning a simple line equation, we're going to learn patterns in 28x28 pixel images!

---

## 2. Run the Full MNIST Example Together

### Goals

* Show a working neural network that classifies handwritten digits.
* Demonstrate the power of deep learning on real image data.
* Let you see the "big picture" before breaking down the components.

We're going to train a neural network to look at images of handwritten digits (like 0, 1, 2, 3, 4, 5, 6, 7, 8, 9) and correctly identify which digit it is!

The MNIST dataset contains 60,000 training images and 10,000 test images of handwritten digits. Each image is 28x28 pixels in grayscale.

---

### Code Demo: MNIST Digit Classification

Paste this into a Python or Colab notebook, or run the script `session_2_example_code.py`:

```python
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
```

---

### What is Happening Here?

* We're loading the **MNIST dataset** - 60,000 training images of handwritten digits.
* Each image is **28x28 pixels** in grayscale (784 total pixel values).
* Our neural network has **3 layers**: 784 → 128 → 64 → 10 neurons.
* The model learns to map pixel patterns to digit classes (0-9).
* We use **CrossEntropyLoss** for multi-class classification and **Adam optimizer** for training.
* After training, we test on 10,000 unseen images to see how well it generalizes!

---

## 3. Explain the Code

### Goals

* Understand how neural networks work for image classification.
* Learn about the MNIST dataset and data loading.
* Understand the difference between regression and classification.
* Build intuition for multi-layer neural networks and activation functions.

---

### Code Walkthrough

---

### 1. **Data Loading and Preprocessing**

```python
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

* **transforms.ToTensor()** converts images from PIL format (0-255 integers) to PyTorch tensors with values scaled to 0.0-1.0 range. This is perfect for neural networks since they work better with smaller, normalized values.
* **MNIST dataset** is automatically downloaded the first time you run this.
* **DataLoader** organizes data into batches and handles shuffling for better training.

---

### 2. **Neural Network Architecture**

```python
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)
```

* **Flatten** converts 28x28 images into 784-length vectors.
* **Linear layers** (`fc1`, `fc2`, `fc3`) are fully connected layers that learn patterns.
* **ReLU activation** adds non-linearity - without it, the network would just be linear regression!
* **Output layer** has 10 neurons (one for each digit 0-9).

---

### 3. **Loss Function and Classification**

```python
criterion = nn.CrossEntropyLoss()
```

* **CrossEntropyLoss** is used for multi-class classification (unlike MSE for regression).
* It measures how far off the predicted probabilities are from the true class.
* The model outputs 10 values (logits), and we take the highest one as the prediction.

---

### 4. **Training Loop Differences**

```python
outputs = model(data)
loss = criterion(outputs, targets)
_, predicted = torch.max(outputs.data, 1)
correct += (predicted == targets).sum().item()
```

* Instead of predicting continuous values, we predict **class probabilities**.
* We use `torch.max()` to find the class with highest probability.
* We track **accuracy** (% correct) in addition to loss.

---

### 5. **Evaluation**

* We use `model.eval()` and `torch.no_grad()` during testing to disable gradient computation.
* We test on completely unseen data to measure **generalization**.

---

### Key Differences from Session 1

* **Multi-class classification** instead of regression
* **Multi-layer networks** with activation functions
* **Image data** instead of simple numbers
* **Batch processing** of data
* **Accuracy metrics** alongside loss

---

## 4. Questions and Guided Hands-On Practice

### Goals

* Reinforce understanding of neural network concepts.
* Practice modifying network architecture and hyperparameters.
* Develop intuition for how different choices affect performance.
* Build confidence with PyTorch's neural network modules.

---

### **Discussion & Questions**

Start by asking questions to check understanding:

* "What's the difference between this neural network and our linear model from Session 1?"
* "Why do we need ReLU activation functions?"
* "What does each number in the output layer represent?"
* "How is CrossEntropyLoss different from MSELoss?"
* "Why do we flatten the 28x28 images?"
* "What happens when we convert image pixels from 0-255 to 0.0-1.0?"

Give time to think and discuss each concept.

**Extra Discussion on Data Preprocessing:**
Let's understand why we preprocess the image data:

```python
# Let's see what happens to pixel values:
sample_pixel_original = 128  # Gray pixel (middle brightness)
sample_pixel_after_totensor = 128 / 255  # ≈ 0.5

print(f"Original pixel value: {sample_pixel_original}")
print(f"After ToTensor: {sample_pixel_after_totensor:.3f}")
```

This simple 0-1 normalization helps because:
- Neural networks work better with smaller numbers (0-1 vs 0-255)
- Gradients are more stable when inputs aren't too large
- It's easier for the network to learn patterns in this range
- All pixels are now on the same scale

---

### **Guided Coding Exercises**

Work through these tasks together, letting you experiment and observe the effects:

---

### Task 1: **Modify Network Architecture**

* Change the hidden layer sizes (try 256, 32, or add more layers).
* Ask: "How do you think this will affect the model's ability to learn?"
* Train for a few epochs and compare accuracy.

```python
# Example: Bigger network
self.fc1 = nn.Linear(28 * 28, 256)  # Increased from 128
self.fc2 = nn.Linear(256, 128)      # Added bigger second layer
self.fc3 = nn.Linear(128, 10)
```

---

### Task 2: **Experiment with Learning Rate**

* Try different learning rates: 0.01, 0.0001, 0.1
* Ask: "What do you expect to happen with a very high or very low learning rate?"
* Observe training curves and convergence speed.

---

### Task 3: **Add Dropout for Regularization**

* Add dropout layers to prevent overfitting:

```python
self.dropout = nn.Dropout(0.2)

# In forward method:
x = self.relu1(x)
x = self.dropout(x)  # Add after activation
```

* Discuss: "What is overfitting and how might dropout help?"

---

### Task 4: **Visualize Learned Features**

* Add code to visualize what the first layer learns:

```python
# Visualize first layer weights
first_layer_weights = model.fc1.weight.data
plt.figure(figsize=(10, 8))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(first_layer_weights[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.suptitle('First Layer Learned Features')
plt.show()
```

---

### Task 5: **Confusion Matrix**

* Create a confusion matrix to see which digits are confused with each other:

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Collect all predictions and true labels
all_predictions = []
all_labels = []

model.eval()
with torch.no_grad():
    for data, targets in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())

# Create confusion matrix
cm = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

---

### Tips During Practice:

* Encourage experimentation: "What happens if we try...?"
* Ask for predictions before running code: "What do you think will happen?"
* Discuss trade-offs: bigger networks vs. training time, accuracy vs. complexity
* Celebrate improvements and learn from setbacks

---

## 5. Independent Challenge

### Goal

Build and train a convolutional neural network (CNN) for MNIST classification - a more advanced architecture that's specifically designed for image data.

---

### Task

Create a CNN that uses convolutional layers instead of just fully connected layers. CNNs are better for images because they can detect local patterns like edges and shapes.

You'll:

* Build a CNN with convolutional and pooling layers
* Train it on MNIST
* Compare performance with the fully connected network
* Visualize the learned convolutional filters

---

### Starter Code

```python
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
```

---

### Challenge Hints

1. **Convolutional Layer**: `nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)`
2. **Pooling Layer**: `nn.MaxPool2d(kernel_size=2, stride=2)`
3. **Calculate sizes**: After each conv/pool layer, track the spatial dimensions
4. **Typical CNN architecture**: Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → FC → ReLU → FC

---

### Expected Results

A well-designed CNN should achieve 98%+ accuracy on MNIST, better than the fully connected network!

---

### Bonus Tasks

* Add batch normalization: `nn.BatchNorm2d()`
* Try different kernel sizes and numbers of filters
* Implement data augmentation with random rotations/translations
* Compare training time between CNN and fully connected network

---

## 6. Wrap-Up + Homework

### Recap

Let's review what we learned today:

* How neural networks classify images vs. fitting lines to data
* The MNIST dataset and working with real image data
* Multi-layer networks with activation functions (ReLU)
* CrossEntropyLoss for classification vs. MSELoss for regression
* Data loaders and batch processing
* Evaluation metrics: accuracy alongside loss
* The difference between training and testing

### Questions for Reflection

* What makes a neural network "deep"?
* Why are activation functions necessary?
* How is image classification different from the linear regression we did last time?
* What did you notice about the training curves?

### Homework Projects

Choose one or more of these projects to deepen your understanding:

---

#### 1. **Fashion-MNIST Classification**

* Replace MNIST with Fashion-MNIST (clothing items instead of digits)
* Use the same network architecture
* Compare: Is it harder or easier than digit classification? Why?

```python
# Just change the dataset:
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
```

---

#### 2. **Hyperparameter Exploration**

* Systematically test different combinations:
  - Learning rates: [0.1, 0.01, 0.001, 0.0001]
  - Network sizes: [64, 128, 256, 512] neurons
  - Number of layers: [1, 2, 3, 4] hidden layers
* Create a table or chart showing which combinations work best
* Document your findings!

---

#### 3. **Build Your Own Dataset**

* Create a simple dataset by drawing digits yourself or using a drawing app
* Save them as 28x28 pixel images
* Test your trained MNIST model on your own handwriting
* See how well it generalizes to your writing style!

---

#### 4. **Error Analysis**

* Find the images that your model gets wrong most often
* Plot them and look for patterns
* Are certain digits consistently confused with others?
* Try to understand why the model makes these mistakes

---

#### 5. **Convolutional Neural Network (if not completed in class)**

* Complete the CNN implementation from the independent challenge
* Compare CNN vs. fully connected network performance
* Visualize what the convolutional filters learn
* Experiment with different CNN architectures

---

## Additional Resources

### Datasets to Explore

* **CIFAR-10**: 32x32 color images with 10 classes (airplanes, cars, birds, etc.)
* **Fashion-MNIST**: 28x28 grayscale images of clothing items
* **EMNIST**: Extended MNIST with letters and digits

### Useful PyTorch Functions

* `torch.nn.Conv2d()` - Convolutional layers
* `torch.nn.BatchNorm2d()` - Batch normalization  
* `torch.nn.Dropout()` - Regularization
* `torchvision.transforms` - Data augmentation

### Next Session Preview

In Session 3, we'll explore:
* Convolutional Neural Networks in depth
* Working with color images (CIFAR-10)
* Transfer learning and pre-trained models
* More advanced architectures like ResNet

---

## Installation Requirements

Make sure you have these packages installed from Session 1:

```bash
pip install torch torchvision torchaudio
pip install matplotlib
pip install scikit-learn
pip install seaborn  # For confusion matrix visualization
```

### GPU Support (Optional)

If you have a CUDA-compatible GPU:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

You can check if GPU is available:

```python
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
```