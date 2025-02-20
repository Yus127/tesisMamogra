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

        # Convert all text labels to numerical labels at initialization
        self.encoded_labels = self.label_encoder.transform(all_labels)
        
        print(f"Found {len(self.label_encoder.classes_)} unique classes")
        print("Class mapping:")
        for i, label in enumerate(self.label_encoder.classes_):
            print(f"  {i}: {label}")
        
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
            "label": self.encoded_labels[idx]
        }

    def get_class_mapping(self):
        """Returns a dictionary mapping numerical labels to their text descriptions"""
        return {i: label for i, label in enumerate(self.label_encoder.classes_)}
        
    def get_num_classes(self):
        """Returns the total number of unique classes"""
        return len(self.label_encoder.classes_)
    
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
        num_classes: int,
        class_mapping: dict,

        learning_rate: float = 0.0001,
        feature_learning_rate: float = 0.00001,
        pretrained: bool = True
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['class_mapping'])

        # Store number of classes
        self.num_classes = num_classes
        self.class_mapping = class_mapping

        
        # Different learning rates for feature extraction and new layers
        self.learning_rate = learning_rate
        self.feature_learning_rate = feature_learning_rate

        # Model initialization
        self.model = models.vgg16(pretrained=pretrained)
        
        # Unfreeze all layers for fine-tuning
        for param in self.model.parameters():
            param.requires_grad = True
            
        # Modify the classifier
        self.model.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            #nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            #nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

        # Initialize the new classifier layers
        for m in self.model.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.4)
                nn.init.zeros_(m.bias)

        # Metrics
        self.train_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", 
            num_classes=self.num_classes
        )
        self.val_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", 
            num_classes=self.num_classes
        )
        self.test_accuracy = torchmetrics.classification.Accuracy(
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
        return self.model(x)


    def _common_step(self, batch, batch_idx):
        # Get batch
        image, labels = batch['image'], batch['label']
        
        # Normalize images if not already normalized
        if image.max() > 1.0:
            image = image / 255.0
        #image = self.normalize(image)
       
        return image, labels

    def _log_confusion_matrix(self, stage):
        if not self.metrics_updated:
            return
            
        conf_matrix = self.confusion_matrix.compute()
        
        # Get class labels from mapping
        labels = [self.class_mapping[i] for i in range(self.num_classes)]
        
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix.cpu().numpy(),
            annot=True,
            fmt='g',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{stage.capitalize()} Confusion Matrix')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        self.logger.experiment.add_figure(
            f'{stage}_confusion_matrix',
            fig,
            self.current_epoch
        )
        plt.close()

    def _log_per_class_metrics(self, stage):
        # Compute metrics
        per_class_acc = self.per_class_accuracy.compute()
        f1_scores = self.f1_score.compute()
        
        # Get class labels from mapping
        labels = [self.class_mapping[i] for i in range(self.num_classes)]
        
        # Log metrics to tensorboard as scalars for each class
        for idx, class_name in enumerate(labels):
            acc_value = per_class_acc[idx].item()
            f1_value = f1_scores[idx].item()
            
            self.log(f'{stage}/acc_class_{class_name}', acc_value, on_epoch=True)
            self.log(f'{stage}/f1_class_{class_name}', f1_value, on_epoch=True)
        
        # Calculate mean metrics
        mean_acc = torch.mean(per_class_acc).item()
        mean_f1 = torch.mean(f1_scores).item()
        
        self.log(f'{stage}/mean_per_class_acc', mean_acc, on_epoch=True)
        self.log(f'{stage}/mean_f1_score', mean_f1, on_epoch=True)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        x = np.arange(len(labels))
        
        # Plot accuracies
        bars1 = ax1.bar(x, per_class_acc.cpu().numpy(), color='skyblue')
        ax1.axhline(y=mean_acc, color='red', linestyle='--', label=f'Mean: {mean_acc:.3f}')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'{stage.capitalize()} Per-Class Accuracy')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        # Plot F1 scores
        bars2 = ax2.bar(x, f1_scores.cpu().numpy(), color='lightgreen')
        ax2.axhline(y=mean_f1, color='red', linestyle='--', label=f'Mean: {mean_f1:.3f}')
        ax2.set_ylabel('F1 Score')
        ax2.set_title(f'{stage.capitalize()} Per-Class F1 Scores')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Log figure to tensorboard
        self.logger.experiment.add_figure(
            f'{stage}/per_class_metrics',
            fig,
            self.current_epoch
        )
        plt.close()

    def training_step(self, batch, batch_idx):
        image, labels = self._common_step(batch, batch_idx)
        
        # Compute logits
        logits = self(image)

        # Compute loss (removed label smoothing and L2 regularization)
        loss = F.cross_entropy(logits, labels)
        
        # Log training metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        self.log(
            name='train_batch_loss',
            value=loss,
            batch_size=image.size(0),
            on_step=True,
            on_epoch=False
        )
        
        # Update metrics
        #_predictions = logits.softmax(dim=-1).argmax(dim=-1)
        _predictions = logits.argmax(dim=1)

        self.train_accuracy.update(_predictions, labels)
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
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        
        # Update metrics
        #_predictions = logits.softmax(dim=-1).argmax(dim=-1)
        _predictions = logits.argmax(dim=-1)

        self.val_accuracy.update(_predictions, labels)
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
        #_predictions = logits.softmax(dim=-1).argmax(dim=-1)
        _predictions = logits.argmax(dim=-1)

        self.test_accuracy.update(_predictions, labels)
        self.confusion_matrix.update(_predictions, labels)
        self.per_class_accuracy.update(_predictions, labels)
        self.f1_score.update(_predictions, labels)
        self.metrics_updated = True

        return loss

    def on_train_epoch_end(self):
        if not self.metrics_updated:
            return
            
        train_acc = self.train_accuracy.compute()
        self.log('train_acc', train_acc, on_epoch=True, prog_bar=True)
        self.train_accuracy.reset()

        
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
            
        val_acc = self.val_accuracy.compute()
        self.log('val_acc', val_acc, on_epoch=True, prog_bar=True)
        
        self._log_confusion_matrix('val')
        self._log_per_class_metrics('val')
        
        self.val_accuracy.reset()
        self.train_accuracy.reset()
        
        self.confusion_matrix.reset()
        self.per_class_accuracy.reset()
        self.f1_score.reset()
        self.metrics_updated = False

    def on_test_epoch_end(self):
        if not self.metrics_updated:
            return
            
        test_acc = self.test_accuracy.compute()
        self.log('test_acc', test_acc, on_epoch=True, prog_bar=True)
        
        
        self._log_confusion_matrix('test')
        self._log_per_class_metrics('test')
        
        self.test_accuracy.reset()
        self.confusion_matrix.reset()
        self.per_class_accuracy.reset()
        self.f1_score.reset()
        self.metrics_updated = False

    def configure_optimizers(self):
        # Separate parameter groups for feature extractor and classifier
        feature_params = []
        classifier_params = []
        
        # Split parameters into feature extractor and classifier groups
        for name, param in self.model.named_parameters():
            if "classifier" in name:
                classifier_params.append(param)
            else:
                feature_params.append(param)
        
        # Create optimizer with different learning rates
        optimizer = torch.optim.Adam([
            {'params': feature_params, 'lr': self.feature_learning_rate},
            {'params': classifier_params, 'lr': self.learning_rate}
        ])
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=0.5,
            patience=5,
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
    dataset = ComplexMedicalDataset(
        data_dir=data_dir,
        train=True
    )
    
    # Get unique classes from the dataset
    num_classes = dataset.get_num_classes()
    class_mapping = dataset.get_class_mapping()

    
    # Initialize datamodule with these transforms
    datamodule = MyDatamodule(
        data_dir = data_dir,
        transforms = transforms_dict,
        batch_size = batch_size,
        num_workers = num_workers
    )
    
    # Initialize model with the unique classes
    model = VGG16Custom(
        num_classes=num_classes,
        class_mapping=class_mapping,
        learning_rate=learning_rate,
        feature_learning_rate=0.00001
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
        logger=L.loggers.TensorBoardLogger(save_dir='logging_tests',name='linear_probe',version = "4_balanced_no_augmentation_vgg_no_norm_v5",default_hp_metric=False))
    
    return trainer, model, datamodule

if __name__ == "__main__":
    class_descriptions = [
        "Fatty predominance",
        "Characterized by scattered areas of pattern density",
        "Heterogeneously dense",
        "Extremely dense"
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
    
 #######
            
