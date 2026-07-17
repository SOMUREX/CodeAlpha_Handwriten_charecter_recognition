import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Classes: 10 digits (0-9) + 26 uppercase letters (A-Z) = 36 classes
EMNIST_CLASSES = [str(i) for i in range(10)] + [chr(i) for i in range(ord('A'), ord('Z') + 1)]
NUM_CLASSES = len(EMNIST_CLASSES)

class SyntheticCharacterDataset(Dataset):
    def __init__(self, samples_per_class=1000):
        self.classes = EMNIST_CLASSES
        self.fonts_dir = "C:\\Windows\\Fonts"
        self.font_files = [
            'arial.ttf', 'calibri.ttf', 'comic.ttf', 'segoepr.ttf', 
            'times.ttf', 'cour.ttf', 'segoesc.ttf', 'tahoma.ttf', 
            'verdana.ttf', 'consola.ttf'
        ]
        
        self.data = []
        self.labels = []
        
        print(f"Generating synthetic dataset: {NUM_CLASSES} classes, {samples_per_class} samples/class...")
        for label_idx, char in enumerate(self.classes):
            for _ in range(samples_per_class):
                # Choose random font
                font_file = random.choice(self.font_files)
                font_path = os.path.join(self.fonts_dir, font_file)
                
                # Draw character on a temporary 64x64 canvas (allows rotation without clipping)
                size = 64
                img = Image.new('L', (size, size), 0)
                draw = ImageDraw.Draw(img)
                
                # Random font size (between 36 and 48)
                font_size = random.randint(36, 48)
                try:
                    font = ImageFont.truetype(font_path, font_size)
                except Exception:
                    font = ImageFont.load_default()
                
                # Center character alignment
                bbox = draw.textbbox((0, 0), char, font=font)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                
                # Random jitter offset
                offset_x = random.randint(-4, 4)
                offset_y = random.randint(-4, 4)
                draw.text(
                    ((size - w) // 2 - bbox[0] + offset_x, (size - h) // 2 - bbox[1] + offset_y), 
                    char, fill=255, font=font
                )
                
                # Random rotation (-15 to 15 degrees)
                angle = random.randint(-15, 15)
                img = img.rotate(angle, resample=Image.BICUBIC)
                
                # Crop drawing bounding box and pad to square
                bbox = img.getbbox()
                if bbox is not None:
                    cropped = img.crop(bbox)
                    cw, ch = cropped.size
                    max_dim = max(cw, ch)
                    
                    square_img = Image.new('L', (max_dim, max_dim), 0)
                    square_img.paste(cropped, ((max_dim - cw) // 2, (max_dim - ch) // 2))
                    
                    # Add 15% margins padding to match inference pipeline
                    border = int(max_dim * 0.15)
                    img = ImageOps.expand(square_img, border=border, fill=0)
                
                # Resize to standard 28x28
                img = img.resize((28, 28), Image.Resampling.LANCZOS)
                
                # Save numpy array
                self.data.append(np.array(img, dtype=np.uint8))
                self.labels.append(label_idx)
                
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        print("Dataset generation completed!")
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        
        # Convert to float and normalize standard MNIST range
        img_float = img.astype(np.float32) / 255.0
        img_normalized = (img_float - 0.1307) / 0.3081
        
        # Convert to tensor shape (1, 28, 28)
        img_tensor = torch.tensor(img_normalized).unsqueeze(0)
        return img_tensor, label

class EMNISTCNN(nn.Module):
    def __init__(self, num_classes=36):
        super(EMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(-1, 64 * 14 * 14)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def main():
    # Instantiate dataset (1000 samples per class)
    full_dataset = SyntheticCharacterDataset(samples_per_class=1000)
    
    # Split train/test (80/20)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    model = EMNISTCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 4
    print(f"Training CNN on synthetic digits + alphabets dataset for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Train Acc: {100. * correct / total:.2f}%")
                
        # Epoch validation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
                
        print(f"--- Epoch [{epoch+1}/{epochs}] Val Loss: {test_loss/len(test_loader):.4f}, Val Acc: {100. * test_correct / test_total:.2f}% ---")
        
    # Export ONNX model
    print("Exporting model to ONNX format...")
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28, device=device)
    onnx_path = "model.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model successfully saved to {onnx_path}!")

if __name__ == "__main__":
    main()
