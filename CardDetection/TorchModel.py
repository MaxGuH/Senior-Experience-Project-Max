# this file is used to train a model to classify the cards in the templates folder and the compTemplates folder
# the model is then used to classify a single card image provided by the user
# the model is trained using a resnet18 model from the torchvision library



import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np

# 1. Data Augmentation and Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
])

# 2. Dataset and DataLoader
class CardDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.label_map = {
    'Ace_C': 0, 'Two_C': 1, 'Three_C': 2, 'Four_C': 3, 'Five_C': 4,
    'Six_C': 5, 'Seven_C': 6, 'Eight_C': 7, 'Nine_C': 8, 'Ten_C': 9,
    'Jack_C': 10, 'Queen_C': 11, 'King_C': 12,
    'Ace_D': 13, 'Two_D': 14, 'Three_D': 15, 'Four_D': 16, 'Five_D': 17,
    'Six_D': 18, 'Seven_D': 19, 'Eight_D': 20, 'Nine_D': 21, 'Ten_D': 22,
    'Jack_D': 23, 'Queen_D': 24, 'King_D': 25,
    'Ace_H': 26, 'Two_H': 27, 'Three_H': 28, 'Four_H': 29, 'Five_H': 30,
    'Six_H': 31, 'Seven_H': 32, 'Eight_H': 33, 'Nine_H': 34, 'Ten_H': 35,
    'Jack_H': 36, 'Queen_H': 37, 'King_H': 38,
    'Ace_S': 39, 'Two_S': 40, 'Three_S': 41, 'Four_S': 42, 'Five_S': 43,
    'Six_S': 44, 'Seven_S': 45, 'Eight_S': 46, 'Nine_S': 47, 'Ten_S': 48,
    'Jack_S': 49, 'Queen_S': 50, 'King_S': 51,
    'Joker': 52, 'Back': 53
}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        # Extract the card name from the filename
        card_name = os.path.basename(img_path).split('.')[0]
        label = self.label_map[card_name]  # Map the card name to an integer label
        return image, label


image_paths = [os.path.join("CardDetection\\Templates", img) for img in os.listdir("CardDetection\\Templates")]
comp_image_paths = [os.path.join("CardDetection\\CompTemplates", img) for img in os.listdir("CardDetection\\CompTemplates")]
image_paths.extend(comp_image_paths)

train_paths, val_paths = train_test_split(image_paths, test_size=0.2, random_state=42)

train_dataset = CardDataset(train_paths, transform=transform)
val_dataset = CardDataset(val_paths, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 3. Model Definition
class CardClassifier(nn.Module):
    def __init__(self):
        super(CardClassifier, self).__init__()
        resnet = models.resnet18(pretrained=True)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # Change to 1 input channel
        num_features = resnet.fc.in_features
        resnet.fc = nn.Linear(num_features, 54)  # 54 classes for 54 cards
        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)



model = CardClassifier()

# 4. Training and Validation
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



best_val_accuracy = 0.0
best_epoch = 0
best_model_state = None
num_epochs = 50
for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    # Validation
    misclassified = []
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Track misclassified examples
            misclassified_indices = (predicted != labels).nonzero(as_tuple=False).squeeze(dim=-1)
            for idx in misclassified_indices:
                misclassified.append((images[idx], predicted[idx], labels[idx]))
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total

# Update and save the best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1
        best_model_state = model.state_dict()
        torch.save(best_model_state, 'test.pth')  # Save the best model state

    # Print epoch results
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

    
# Analyze misclassified examples
print(f'Total misclassified examples: {len(misclassified)}')
for image, pred, label in misclassified[:10]:  # Show the first 10 misclassified examples
    print(f'Predicted: {pred.item()}, Actual: {label.item()}')
    # You can also display the image using matplotlib or another library


# 5. Testing
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = transform(image)  # Apply the same transformations as the training data
    return image.unsqueeze(0)  # Add batch dimension

test_image_path = "detectedCard5.jpg"  # Replace with your own image and path*********
test_image = preprocess_image(test_image_path)
model.eval()
with torch.no_grad():
    output = model(test_image)
    predicted_class = torch.argmax(output, dim=1)
    print(f'Predicted class: {predicted_class.item()}')

    # Analyze misclassified examples
print(f'Total misclassified examples: {len(misclassified)}')
for image, pred, label in misclassified[:10]:  # Show the first 10 misclassified examples
    print(f'Predicted: {pred.item()}, Actual: {label.item()}')

