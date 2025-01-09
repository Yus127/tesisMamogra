import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
import os
import json
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class MammogramDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_training=True):
        """
        Initialize dataset from a JSON file containing mammogram data
        """
        self.base_path = os.path.dirname(data_dir)
        
        # Load and parse JSON data
        with open(data_dir, 'r') as f:
            data_list = json.load(f)

        # Base transformation pipeline
        base_transforms = [
            transforms.Resize((224, 224)),
        ]
        
        # Add augmentation for training
        if is_training:
            base_transforms.extend([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(
                    degrees=10, 
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1)
                ),
            ])
        else:
            base_transforms.append(transforms.CenterCrop(224))
            
        base_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform = transforms.Compose(base_transforms)
        
        # Process data samples
        self.samples = []
        self.class_mapping = {
            'Characterized by scattered areas of pattern density': 0,
            'Extremely dense': 1,
            'Heterogeneously dense': 2,
            'Fatty predominance': 3,
            'Moderately dense': 4
        }
        
        # Add debugging counters
        total_items = 0
        missing_paths = 0
        missing_class = 0
        invalid_class = 0
        
        for item in data_list:
            total_items += 1
            for image_name, content in item.items():
                print(f"Processing item: {image_name}")
                print(f"Content: {content}")
                
                if not isinstance(content, dict):
                    print(f"Warning: content for {image_name} is not a dictionary")
                    continue
                    
                if 'image_paths' not in content:
                    print(f"Warning: no image_paths for {image_name}")
                    missing_paths += 1
                    continue
                    
                if 'report' not in content:
                    print(f"Warning: no density_class for {image_name}")
                    missing_class += 1
                    continue
                
                density_class = content['report'].strip()
                if density_class not in self.class_mapping:
                    print(f"Warning: invalid density_class '{density_class}' for {image_name}")
                    print(f"Valid classes are: {list(self.class_mapping.keys())}")
                    invalid_class += 1
                    continue
                
                # Verify image path exists
                image_path = content['image_paths'][0]
                full_path = os.path.join(self.base_path, image_path)
                if not os.path.exists(full_path):
                    print(f"Warning: image file not found at {full_path}")
                    continue
                
                self.samples.append({
                    'image_name': image_name,
                    'image_path': image_path,
                    'label': self.class_mapping[density_class]
                })
        
        print("\nDataset Loading Summary:")
        print(f"Total items processed: {total_items}")
        print(f"Missing image paths: {missing_paths}")
        print(f"Missing density class: {missing_class}")
        print(f"Invalid density class: {invalid_class}")
        print(f"Valid samples loaded: {len(self.samples)}")
        
        if len(self.samples) == 0:
            raise ValueError("No valid samples were loaded. Please check the data format and paths.")

    def process_image(self, image_path):
        """
        Process a single image, handling both regular images and TIFF/TIF files
        """
        try:
            full_path = os.path.join(self.base_path, image_path)
            
            # Open the image
            img = Image.open(full_path)
            
            # Handle TIFF/TIF specific processing
            if img.format in ['TIFF', 'TIF']:
                # Convert to numpy array
                img_array = np.array(img)
                
                # Handle different bit depths
                if img_array.dtype != np.uint8:
                    # Normalize to 8-bit range
                    img_array = ((img_array - img_array.min()) * (255.0 / (img_array.max() - img_array.min()))).astype(np.uint8)
                
                # Handle different channel configurations
                if len(img_array.shape) == 2:  # Single channel
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif len(img_array.shape) == 3:
                    if img_array.shape[2] == 1:  # Single channel in 3D
                        img_array = np.concatenate([img_array] * 3, axis=2)
                    elif img_array.shape[2] == 4:  # RGBA
                        img_array = img_array[:, :, :3]  # Take only RGB channels
                
                # Convert back to PIL
                img = Image.fromarray(img_array)
            else:
                # For non-TIF images, convert to RGB
                img = img.convert('RGB')
            
            # Apply the transformation pipeline
            processed_img = self.transform(img)
            return processed_img
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return torch.zeros((3, 224, 224))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = self.process_image(sample['image_path'])
        return image, sample['label']

def create_model(num_classes=5):
    # Load pretrained ConvNeXt
    model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
    
    # Modify the classifier
    model.classifier[2] = nn.Linear(1024, num_classes)
    
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            print("labels")
            print(labels)
            print("predicted")
            print(predicted)
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pth')
            print(f'Best model saved with validation accuracy: {val_acc:.2f}%')

def main():
    torch.manual_seed(42)
    
    # Hyperparameters
    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-4
    
    # Create datasets
    train_dataset = MammogramDataset('/Users/YusMolina/Documents/tesis/biomedCLIP/data/datosMex/images/train.json', is_training=True)
    val_dataset = MammogramDataset('/Users/YusMolina/Documents/tesis/biomedCLIP/data/datosMex/images/test.json', is_training=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model, criterion, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model()
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

if __name__ == '__main__':
    main()