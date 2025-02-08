# is the multiclass classifier
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
import cv2

import torch
import torch.nn as nn
import lightning as L
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from tqdm import tqdm
import torchvision.transforms as transforms

from sklearn.preprocessing import LabelEncoder
import numpy as np

class ComplexMedicalDataset(Dataset):
    def __init__(self, data_dir: str, train: bool = True, transform=None):
        super(ComplexMedicalDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.label_encoder = LabelEncoder()
        
        # Load the appropriate JSON file
        json_file = "train.json" if train else "test.json"
        if not os.path.exists(os.path.join(self.data_dir, json_file)):
            raise FileNotFoundError(f"{json_file} not found in the data directory.")
            
        with open(os.path.join(data_dir, json_file), 'r') as f:
            self.data = json.load(f)
        
        # Extract all unique labels and fit the encoder
        all_labels = [item['report'] for item in self.data]
        self.label_encoder.fit(all_labels)
        
        print(f"Found {len(self.label_encoder.classes_)} unique classes: {self.label_encoder.classes_}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        
        # Load image
        image_path = item["filename"]
        image = cv2.imread(os.path.join(self.data_dir, image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found at {os.path.join(self.data_dir, image_path)}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        # Convert text label to numerical
        label = self.label_encoder.transform([item['report']])[0]
        
        return {
            "image": image,
            "text": label  # Now returns a numerical label
        }
    
    def get_label_encoder(self):
        return self.label_encoder


class MyDatamodule(L.LightningDataModule):
    def __init__(self, data_dir: str, transforms: dict, batch_size: int = 32, num_workers: int = 1):
        super(MyDatamodule, self).__init__()
        self.data_dir = data_dir
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Initialize datasets as None
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        
        # Print initialization info
        print(f"Initialized DataModule with data_dir: {data_dir}")

    def setup(self, stage=None):
        print(f"\nSetting up data for stage: {stage}")
        print(f"Looking for data in: {self.data_dir}")
        
        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # List contents of data directory
        print("\nContents of data directory:")
        for item in os.listdir(self.data_dir):
            print(f"- {item}")
        
        try:
            print("\nAttempting to load training data...")
            training_data = ComplexMedicalDataset(
                data_dir=self.data_dir,
                train=True,
                transform=self.transforms['train']
            )
            print(f"Successfully loaded training data with {len(training_data)} samples")
            
            # Calculate split sizes
            total_size = len(training_data)
            train_size = int(0.8 * total_size)
            val_size = total_size - train_size
            print(f"\nSplitting data into {train_size} training and {val_size} validation samples")
            
            # Create the split
            self.train_dataset, self.validation_dataset = torch.utils.data.random_split(
                training_data, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)  # For reproducibility
            )
            print("Successfully created train/validation split")
            
        except FileNotFoundError as e:
            print(f"\nError loading training data: {e}")
            self.train_dataset = None
            self.validation_dataset = None
        except Exception as e:
            print(f"\nUnexpected error loading training data: {e}")
            self.train_dataset = None
            self.validation_dataset = None
            
        try:
            print("\nAttempting to load test data...")
            self.test_dataset = ComplexMedicalDataset(
                data_dir=self.data_dir,
                train=False,
                transform=self.transforms['test']
            )
            print(f"Successfully loaded test data with {len(self.test_dataset)} samples")
        except FileNotFoundError as e:
            print(f"\nError loading test data: {e}")
            self.test_dataset = None
        except Exception as e:
            print(f"\nUnexpected error loading test data: {e}")
            self.test_dataset = None
            
        # Final status report
        print("\nDataset setup complete:")
        print(f"Training samples: {len(self.train_dataset) if self.train_dataset else 'None'}")
        print(f"Validation samples: {len(self.validation_dataset) if self.validation_dataset else 'None'}")
        print(f"Test samples: {len(self.test_dataset) if self.test_dataset else 'None'}")

    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("No training dataset found. Check the setup logs for details.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        if self.validation_dataset is None:
            raise ValueError("No validation dataset found. Check the setup logs for details.")
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("No test dataset found. Check the setup logs for details.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

import torch
import torch.nn as nn
import lightning as L
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from tqdm import tqdm
import torchvision.transforms as transforms

def create_model(num_classes=5):
    # Load pretrained ConvNeXt
    model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
    
    # Modify the classifier
    model.classifier[2] = nn.Linear(1024, num_classes)
    
    return model

class MedicalTrainer(L.LightningModule):
    def __init__(self, num_classes=5, learning_rate=1e-4):
        super().__init__()
        self.model = create_model(num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        
        # Metrics
        self.train_acc = 0.0
        self.val_acc = 0.0
        self.best_val_acc = 0.0
        
        # Label encoding
        self.label_encoder = None  # Will be set from the datamodule
    
    def set_label_encoder(self, label_encoder):
        self.label_encoder = label_encoder
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images = batch['image']
        labels = torch.tensor(batch['text'], device=self.device)  # Convert list to tensor
        
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        self.train_acc = 100. * correct / labels.size(0)
        
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images = batch['image']
        labels = torch.tensor(batch['text'], device=self.device)  # Convert list to tensor
        
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        self.val_acc = 100. * correct / labels.size(0)
        
        self.log('val_loss', loss)
        self.log('val_acc', self.val_acc)
        
        # Save best model
        if self.val_acc > self.best_val_acc:
            self.best_val_acc = self.val_acc
            self.save_model()
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
    
    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'val_acc': self.val_acc,
        }, 'best_model.pth')
        print(f'Best model saved with validation accuracy: {self.val_acc:.2f}%')

def main():
    torch.manual_seed(42)
    
    # Hyperparameters
    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-4
    
    # Define transforms
    transforms_dict = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    
    # Create data module
    data_module = MyDatamodule(
        data_dir='/mnt/Pandora/Datasets/MamografiasMex/4kimages/',  # data directory
        transforms=transforms_dict,
        batch_size=batch_size,
        num_workers=4
    )
    
    # Set up the data module
    data_module.setup()
    
    # Initialize trainer and model
    trainer = L.Trainer(
        max_epochs=num_epochs,
        accelerator='auto',
        devices=1
    )
    
    model = MedicalTrainer(num_classes=4, learning_rate=learning_rate)
    
    # Train the model
    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()
