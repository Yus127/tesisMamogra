import os
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_lightning import LightningModule 



class ComplexMedicalDataset(Dataset):
    def __init__(self, data_dir: str, train: bool = True, transform=None):
        super(ComplexMedicalDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.label_encoder = LabelEncoder()
        
        # Load JSON file
        json_file = "train_balanced.json" if train else "test_balanced.json"
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
        full_path = os.path.join(self.data_dir, image_path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found at {full_path}")
        
        image = cv2.imread(full_path)
        if image is None:
            raise ValueError(f"Failed to load image at {full_path}")
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert numpy array to PIL Image
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        # Return the original text label
        return {
            "image": image,
            "text": item['report'] 
        }
    
class MyDatamodule(L.LightningDataModule):
    def __init__(self, data_dir: str, transforms: dict, batch_size: int, num_workers: int):
        super().__init__()
        self.data_dir = data_dir
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        
        print(f"Initialized DataModule with data_dir: {data_dir}")

    def setup(self, stage=None):
        print(f"\nSetting up data for stage: {stage}")
        print(f"Looking for data in: {self.data_dir}")
        
        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        if stage == 'fit' or stage is None:
            try:
                print("\nAttempting to load training data...")
                training_data = ComplexMedicalDataset(
                    data_dir=self.data_dir,
                    train=True,
                    transform=self.transforms['train']
                )
                
                # Calculate split sizes
                total_size = len(training_data)
                train_size = int(0.85 * total_size)
                val_size = total_size - train_size
                print(f"\nSplitting data into {train_size} training and {val_size} validation samples")
                
                # Create the split
                self.train_dataset, self.validation_dataset = torch.utils.data.random_split(
                    training_data, 
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(42)
                )
                
                # Verify the splits were created
                if self.train_dataset is None or self.validation_dataset is None:
                    raise ValueError("Failed to create train/validation split")
                
                print(f"Successfully created train/validation split")
                print(f"Train size: {len(self.train_dataset)}")
                print(f"Validation size: {len(self.validation_dataset)}")
                
            except Exception as e:
                print(f"\nError in setup: {str(e)}")
                raise e

        if stage == 'test' or stage is None:
            try:
                print("\nAttempting to load test data...")
                self.test_dataset = ComplexMedicalDataset(
                    data_dir=self.data_dir,
                    train=False,
                    transform=self.transforms['test']
                )
                print(f"Successfully loaded test data with {len(self.test_dataset)} samples")
            except Exception as e:
                print(f"\nError loading test data: {str(e)}")
                self.test_dataset = None

    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("No training dataset found. Have you called setup()?")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        if self.validation_dataset is None:
            raise ValueError("No validation dataset found. Have you called setup()?")
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("No test dataset found. Have you called setup()?")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


class VGG16Custom(LightningModule):
    def __init__(
        self, 
        class_descriptions: list, 
        learning_rate: float = 0.0001,
        pretrained: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()

        # Store class descriptions and create label encoder
        self.class_text = class_descriptions
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(class_descriptions)

        # Model initialization
        self.model = models.vgg16(pretrained=pretrained)
        
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Get feature dimension (last layer before classifier)
        self.feature_dim = self.model.classifier[-1].in_features
        
        # Remove the last classification layer
        self.model.classifier = self.model.classifier[:-1]

        # Define an index for each class description
        self.num_classes = len(self.class_text)

        # Simplified classifier without dropout
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.num_classes)
        )
        # Initialize the new layer
        nn.init.xavier_uniform_(self.classifier[-1].weight, gain=1.4)
        nn.init.zeros_(self.classifier[-1].bias)

        # Hyperparameters
        self.learning_rate = learning_rate

        # Normalization for pretrained VGG16
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Metrics
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", 
            num_classes=self.num_classes
        )
        self.confusion_matrix = torchmetrics.classification.ConfusionMatrix(
            task="multiclass",
            num_classes=self.num_classes
        )
        self.per_class_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=self.num_classes,
            average=None
        )
        self.f1_score = torchmetrics.classification.F1Score(
            task="multiclass",
            num_classes=self.num_classes,
            average=None
        )
        
        # Initialize metric states tracker
        self.metrics_updated = False

    def forward(self, x):
        features = self.model(x)
        return self.classifier(features)

    def _common_step(self, batch, batch_idx):
        # Get batch
        image, text = batch['image'], batch['text']
        
        # Normalize images if not already normalized
        if image.max() > 1.0:
            image = image / 255.0
        #image = self.normalize(image)
        
        # Convert text labels to indices
        try:
            labels = torch.tensor([self.class_text.index(t) for t in text], 
                                dtype=torch.long,
                                device=self.device)
        except ValueError as e:
            print("Available classes:", self.class_text)
            print("Received labels:", text)
            raise ValueError(f"Label mismatch. Make sure all labels in your data are included in class_descriptions. Error: {e}")
        
        return image, labels

    def _log_confusion_matrix(self, stage):
        if not self.metrics_updated:
            return
            
        conf_matrix = self.confusion_matrix.compute()
        
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix.cpu().numpy(),
            annot=True,
            fmt='g',
            xticklabels=self.class_text,
            yticklabels=self.class_text
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{stage.capitalize()} Confusion Matrix')
        
        self.logger.experiment.add_figure(
            f'{stage}_confusion_matrix',
            fig,
            self.current_epoch
        )
        plt.close()

    def _log_per_class_metrics(self, stage):
        if not self.metrics_updated:
            return
            
        per_class_acc = self.per_class_accuracy.compute()
        f1_scores = self.f1_score.compute()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(self.class_text))
        width = 0.35
        
        ax.bar(x - width/2, per_class_acc.cpu(), width, label='Accuracy')
        ax.bar(x + width/2, f1_scores.cpu(), width, label='F1 Score')
        
        ax.set_ylabel('Score')
        ax.set_title(f'{stage.capitalize()} Per-Class Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_text, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        self.logger.experiment.add_figure(
            f'{stage}_per_class_metrics',
            fig,
            self.current_epoch
        )
        plt.close()
        
        # Log individual metrics
        for idx, class_name in enumerate(self.class_text):
            self.log(f'{stage}_acc_{class_name}', per_class_acc[idx])
            self.log(f'{stage}_f1_{class_name}', f1_scores[idx])

    def training_step(self, batch, batch_idx):
        image, labels = self._common_step(batch, batch_idx)
        
        # Compute logits
        logits = self(image)

        # Compute loss (removed label smoothing and L2 regularization)
        loss = F.cross_entropy(logits, labels)
        
        # Log training metrics
        self.log(
            name='train_loss', 
            value=loss,  
            batch_size=image.size(0),
            on_step=False,
            on_epoch=True
        )
        self.log(
            name='train_batch_loss',
            value=loss,
            batch_size=image.size(0),
            on_step=True,
            on_epoch=False
        )
        
        # Update metrics
        _predictions = logits.softmax(dim=-1).argmax(dim=-1)
        self.accuracy.update(_predictions, labels)
        self.confusion_matrix.update(_predictions, labels)
        self.per_class_accuracy.update(_predictions, labels)
        self.f1_score.update(_predictions, labels)
        self.metrics_updated = True

        return loss

    def validation_step(self, batch, batch_idx):
        image, labels = self._common_step(batch, batch_idx)
        
        # Compute logits
        logits = self(image)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        # Log validation metrics
        self.log(
            name='val_loss',
            value=loss,
            batch_size=image.size(0),
            on_step=False,
            on_epoch=True
        )
        
        # Update metrics
        _predictions = logits.softmax(dim=-1).argmax(dim=-1)
        self.accuracy.update(_predictions, labels)
        self.confusion_matrix.update(_predictions, labels)
        self.per_class_accuracy.update(_predictions, labels)
        self.f1_score.update(_predictions, labels)
        self.metrics_updated = True

        return loss

    def test_step(self, batch, batch_idx):
        image, labels = self._common_step(batch, batch_idx)
        
        # Compute logits
        logits = self(image)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        # Log test metrics
        self.log(
            name='test_loss',
            value=loss,
            batch_size=image.size(0),
            on_step=False,
            on_epoch=True
        )
        
        # Update metrics
        _predictions = logits.softmax(dim=-1).argmax(dim=-1)
        self.accuracy.update(_predictions, labels)
        self.confusion_matrix.update(_predictions, labels)
        self.per_class_accuracy.update(_predictions, labels)
        self.f1_score.update(_predictions, labels)
        self.metrics_updated = True

        return loss

    def on_train_epoch_end(self):
        if not self.metrics_updated:
            return
            
        acc = self.accuracy.compute()
        self.log('train_acc', acc)
        
        self._log_confusion_matrix('train')
        self._log_per_class_metrics('train')
        
        self.accuracy.reset()
        self.confusion_matrix.reset()
        self.per_class_accuracy.reset()
        self.f1_score.reset()
        self.metrics_updated = False

    def on_validation_epoch_end(self):
        if not self.metrics_updated:
            return
            
        acc = self.accuracy.compute()
        self.log('val_acc', acc)
        
        self._log_confusion_matrix('val')
        self._log_per_class_metrics('val')
        
        self.accuracy.reset()
        self.confusion_matrix.reset()
        self.per_class_accuracy.reset()
        self.f1_score.reset()
        self.metrics_updated = False

    def on_test_epoch_end(self):
        if not self.metrics_updated:
            return
            
        acc = self.accuracy.compute()
        self.log('test_acc', acc)
        
        self._log_confusion_matrix('test')
        self._log_per_class_metrics('test')
        
        self.accuracy.reset()
        self.confusion_matrix.reset()
        self.per_class_accuracy.reset()
        self.f1_score.reset()
        self.metrics_updated = False

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.classifier.parameters(),
            lr=self.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=0.5,
            patience=50,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

def get_transforms():
    return {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    }

def setup_training(data_dir: str, batch_size, learning_rate, num_workers):
    # Initialize transforms
    transforms_dict = get_transforms()
    
    # Create a temporary dataset to get all unique classes
    temp_dataset = ComplexMedicalDataset(
        data_dir=data_dir,
        train=True,
        transform=transforms_dict['train']
    )
    
    # Get unique classes from the dataset
    unique_classes = list(set(item['report'] for item in temp_dataset.data))
    print(f"Found {len(unique_classes)} unique classes:", unique_classes)
    
    # Initialize datamodule with these transforms
    datamodule = MyDatamodule(
        data_dir = data_dir,
        transforms = transforms_dict,
        batch_size = batch_size,
        num_workers = num_workers
    )
    
    # Initialize model with the unique classes
    model = VGG16Custom(
        class_descriptions=unique_classes,
        learning_rate=learning_rate
    )
    
    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=20,
        accelerator='auto',
        devices=1,
        callbacks=[
            L.callbacks.ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_top_k=1
            ),
            L.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                mode='min'
            )
        ],
        logger=L.loggers.TensorBoardLogger(save_dir='logging_tests',name='linear_probe',version = "4_balanced_no_augmentation_vgg_no_norm_v3",default_hp_metric=False))
    
    return trainer, model, datamodule

if __name__ == "__main__":
    class_descriptions = [
        "Characterized by scattered areas of pattern density",
        "Fatty predominance",
        "Extremely dense",
        "Heterogeneously dense"
    ]
    
    # Setup training components

    trainer, model, datamodule = setup_training(
        data_dir="/mnt/Pandora/Datasets/MamografiasMex/4kimages/",
        batch_size=64,
        learning_rate=0.0001, 
        num_workers=19
    )
   
    
    # Setup the datamodule
    datamodule.setup(stage='fit')   

    print(f"Train dataset size: {len(datamodule.train_dataset)}")
    print(f"Validation dataset size: {len(datamodule.validation_dataset)}")

    # Train the model
    trainer.fit(model, datamodule=datamodule)

    # Train the model
    trainer.fit(model, datamodule=datamodule)
    
    # Test the model
    trainer.test(model, datamodule=datamodule)
    
    # Save the model
    torch.save(model.state_dict(), 'medical_classifier.pth')
    
 
            
