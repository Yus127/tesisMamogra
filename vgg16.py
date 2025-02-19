import os
import json
import torch
import lightning as L
import pytorch_lightning as L
import torch.nn as nn
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import torchmetrics
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_lightning import LightningModule  # Add this import explicitly


import os
import json
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

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
        
        # Return the original text label instead of encoded version
        return {
            "image": image,
            "text": item['report']  # Return original text label
        }
    
class MyDatamodule(L.LightningDataModule):
    def __init__(self, data_dir: str, transforms: dict, batch_size: int = 32, num_workers: int = 1):
        super().__init__()
        self.data_dir = data_dir
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Initialize datasets as None
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
                #print(f"Successfully loaded training data with {len(training_data)} samples")
                
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

def create_model(num_classes=4):
    # Load pretrained ConvNeXt
    model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
    
    # Modify the classifier
    model.classifier[2] = nn.Linear(1024, num_classes)
    
    return model

class VGG16LinearProbe(LightningModule):
    def __init__(
        self, 
        class_descriptions: list, 
        learning_rate: float = 0.0001,
        weight_decay: float = 0.0, 
        dropout_rate: float = 0.2, 
        l2_lambda: float = 0.00,
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

        # Classifier initialization
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, self.num_classes)
        )
        # Initialize the new layer
        nn.init.xavier_uniform_(self.classifier[-1].weight, gain=1.4)
        nn.init.zeros_(self.classifier[-1].bias)

        # Hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda

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

    def _common_step(self, batch, batch_idx):
        # Get batch
        image, text = batch['image'], batch['text']
        
        # Normalize images if not already normalized
        if image.max() > 1.0:
            image = image / 255.0
        image = self.normalize(image)
        
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


    def _evaluate(self, batch, stage=None):
        image, labels = self._common_step(batch, None)
        
        # Compute logits
        logits = self(image)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        # Update metrics
        _predictions = logits.softmax(dim=-1).argmax(dim=-1)
        self.accuracy.update(_predictions, labels)
        self.confusion_matrix.update(_predictions, labels)
        self.per_class_accuracy.update(_predictions, labels)
        self.f1_score.update(_predictions, labels)
        self.metrics_updated = True

        # Log metrics
        self.log(
            name=f'{stage}_loss',
            value=loss,
            batch_size=image.size(0)
        )

    def training_step(self, batch, batch_idx):
        image, labels = self._common_step(batch, batch_idx)
        
        # Compute logits
        logits = self(image)

        # Compute loss
        loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        if self.l2_lambda > 0.0:
            l2_norm = sum(
                torch.sum(param ** 2) 
                for param in self.classifier.parameters()
            )
            loss = loss + self.l2_lambda * l2_norm
        
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

    def on_train_epoch_end(self):        
        if not self.metrics_updated:
            return
            
        acc = self.accuracy.compute()
        
        self.log('train_acc', acc)
        
        self._log_confusion_matrix('train')
        self._log_per_class_metrics('train')
        
        # Reset epoch loss counter
        self.train_epoch_loss = 0
        
        # Reset metrics
        self.accuracy.reset()
        self.confusion_matrix.reset()
        self.per_class_accuracy.reset()
        self.f1_score.reset()
        self.metrics_updated = False

    def validation_step(self, batch, batch_idx):
        self._evaluate(batch, stage='val')
    
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

    def test_step(self, batch, batch_idx):
        self._evaluate(batch, stage='test')
    
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

    def forward(self, x):
        features = self.model(x)
        return self.classifier(features)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.classifier.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
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
  
import os
import torch
import pytorch_lightning as L
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision import models
import torchmetrics
from torch.utils.data import DataLoader

# First, let's create some basic transforms
def get_transforms():
    return {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    }

# Assuming ComplexMedicalDataset is defined elsewhere, here's how to set everything up:
def setup_training(data_dir: str, batch_size: int = 32, learning_rate: float = 0.0001):
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
        data_dir=data_dir,
        transforms=transforms_dict,
        batch_size=batch_size
    )
    
    # Initialize model with the unique classes
    model = VGG16LinearProbe(
        class_descriptions=unique_classes,
        learning_rate=learning_rate
    )
    
    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=100,
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
        logger=L.loggers.TensorBoardLogger(save_dir='logging_tests',name='linear_probe',version = "4_unbalanced_no_augmentation_vgg",default_hp_metric=False)
    
    return trainer, model, datamodule

# Example usage:
if __name__ == "__main__":
    # Define your classes
    class_descriptions = [
        "Characterized by scattered areas of pattern density",
        "Fatty predominance",
        "Extremely dense",
        "Heterogeneously dense"
    ]
    
    # Setup training components

    trainer, model, datamodule = setup_training(
        data_dir="/mnt/Pandora/Datasets/MamografiasMex/4kimages/",
        batch_size=32,
        learning_rate=0.0001
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
    
    # Optional: Example of loading and using the model for inference
    def predict_image(model, image_tensor):
        model.eval()
        with torch.no_grad():
            # Ensure image is normalized
            if image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0
            image_tensor = model.normalize(image_tensor)
            
            # Get prediction
            logits = model(image_tensor.unsqueeze(0))
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            
            return {
                'class': model.class_text[predicted_class.item()],
                'probabilities': probabilities[0].tolist()
            }
"""
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
        data_dir='/Users/YusMolina/Documents/tesis/biomedCLIP/data/datosMex/images/4kimages',
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
        save_dir='logs',
        name='medical_training',
        default_hp_metric=False
    )

    early_stop_callback = EarlyStopping(
            monitor='val_loss',  # quantity to monitor
            min_delta=0.00,            # minimum change to qualify as an improvement
            patience=10,               # number of epochs with no improvement after which training will be stopped
            verbose=True,              # enable verbose mode
            mode='min'                 # "min" means lower val_loss is better
        )
    
    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=num_epochs,
        accelerator='auto',
        devices=1,
        logger=tensorboard_logger,
        log_every_n_steps=10,
        callbacks=[early_stop_callback]
    )
    
    # Initialize and train model
    # Initialize the model
    model = VGG16LinearProbe(
        class_descriptions=["Characterized by scattered areas of pattern density",
        "Fatty predominance",
        "Extremely dense",
        "Heterogeneously dense"],
        learning_rate=0.0001,
        dropout_rate=0.2
    )

    # Create trainer and train
    trainer = L.Trainer(
        max_epochs=100,
        accelerator='gpu',
        devices=1,
        logger=L.loggers.TensorBoardLogger('logs/')
    )
    trainer.fit(model, train_dataloader, val_dataloader)


    model = MedicalTrainer(num_classes=4, learning_rate=learning_rate)
    model.set_label_encoder(label_encoder)
    trainer.fit(model, data_module)
    
    # Final validation
    trainer.validate(model, data_module)

if __name__ == '__main__':
    main()
"""
