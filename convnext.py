import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
import os
import json
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2
import torch
import lightning as L
import torchvision.transforms as transforms
from lightning.pytorch.loggers import TensorBoardLogger
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
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
            train_size = int(0.85 * total_size)
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
        
        # For confusion matrix
        self.all_preds = []
        self.all_labels = []
        
        # Label encoding
        self.label_encoder = None
        
        # Initialize tensorboard logger
        self.tensorboard = None
    
    def set_label_encoder(self, label_encoder):
        self.label_encoder = label_encoder
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images = batch['image']
        labels = torch.tensor(batch['text'], device=self.device)
        
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        self.train_acc = 100. * correct / labels.size(0)
        
        # Log metrics to tensorboard
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/accuracy', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log learning rate
        self.log('train/learning_rate', self.optimizer.param_groups[0]['lr'], on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images = batch['image']
        labels = torch.tensor(batch['text'], device=self.device)
        
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        self.val_acc = 100. * correct / labels.size(0)
        
        # Store predictions and labels for confusion matrix
        self.all_preds.extend(predicted.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())
        
        # Log metrics to tensorboard
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/accuracy', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        if self.val_acc > self.best_val_acc:
            self.best_val_acc = self.val_acc
            self.save_model()
        
        return loss
    
    def on_validation_epoch_end(self):
        # Calculate and plot confusion matrix
        if len(self.all_preds) > 0:
            conf_matrix = confusion_matrix(self.all_labels, self.all_preds)
            
            # Get class names if label_encoder is available
            if self.label_encoder is not None:
                class_names = self.label_encoder.classes_
            else:
                class_names = [f'Class {i}' for i in range(len(conf_matrix))]
            
            # Plot confusion matrix
            fig = plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names,
                       yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            
            # Log confusion matrix to tensorboard
            self.logger.experiment.add_figure('val/confusion_matrix', fig, self.current_epoch)
            plt.close()
            
            # Calculate per-class accuracy
            per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
            
            # Log per-class accuracy to tensorboard
            for class_name, acc in zip(class_names, per_class_acc):
                self.log(f'val/accuracy_{class_name}', acc * 100)
            
            # Get classification report
            report = classification_report(self.all_labels, self.all_preds,
                                        target_names=class_names,
                                        output_dict=True)
            
            # Log precision, recall, and f1-score for each class
            for class_name in class_names:
                metrics = report[class_name]
                self.log(f'val/precision_{class_name}', metrics['precision'] * 100)
                self.log(f'val/recall_{class_name}', metrics['recall'] * 100)
                self.log(f'val/f1_{class_name}', metrics['f1-score'] * 100)
            
            # Reset storage
            self.all_preds = []
            self.all_labels = []
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return self.optimizer
    
    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'val_acc': self.val_acc,
        }, 'best_model.pth')
        print(f'Best model saved with validation accuracy: {self.val_acc:.2f}%')

def main():
    torch.manual_seed(42)
    
    # Hyperparameters
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.0001
    
    # Define transforms
    transforms_dict = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
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
    
    # Get label encoder from training dataset
    label_encoder = data_module.train_dataset.dataset.get_label_encoder()
    
    # Initialize tensorboard logger
    tensorboard_logger = TensorBoardLogger(
        save_dir='logging_tests',
        name='linear_probe',
        version = "convnext_no_augmentation",

        default_hp_metric=False
    )
    
    # Initialize trainer and model
    trainer = L.Trainer(
        max_epochs=num_epochs,
        accelerator='auto',
        devices=1,
        logger=tensorboard_logger,
        log_every_n_steps=10
    )
    
    model = MedicalTrainer(num_classes=4, learning_rate=learning_rate)
    model.set_label_encoder(label_encoder)
    
    # Train the model
    trainer.fit(model, data_module)
    
    # After training, perform validation to get final confusion matrix
    trainer.validate(model, data_module)

if __name__ == '__main__':
    main()
