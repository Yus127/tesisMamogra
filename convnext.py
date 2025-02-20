import os
import json
import torch
import lightning as L
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


class ComplexMedicalDataset(Dataset):
    def __init__(self, data_dir: str, train: bool = True, transform=None):
        super(ComplexMedicalDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.label_encoder = LabelEncoder()

        # Load the appropriate JSON file
        json_file = "train_4_balanced_v2.json" if train else "test_4_balanced_v2.json"
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
    def __init__(self, data_dir: str, transforms: dict, batch_size: int, num_workers: int):
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
    def __init__(self, num_classes, learning_rate):
        super().__init__()
        self.save_hyperparameters()

        self.model = create_model(num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        
        # Initialize metrics using torchmetrics
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", 
            num_classes=num_classes
        )
        self.confusion_matrix = torchmetrics.classification.ConfusionMatrix(
            task="multiclass",
            num_classes=num_classes
        )
        self.per_class_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=num_classes,
            average=None
        )
        self.f1_score = torchmetrics.classification.F1Score(
            task="multiclass",
            num_classes=num_classes,
            average=None
        )
        
        # Label encoding
        self.label_encoder = None

@@ -252,128 +246,192 @@
        )

        return loss
























    def training_step(self, batch, batch_idx):
        images, labels = self._common_step(batch, batch_idx)

        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # Log metrics
        self.log(
            name='train_loss', 
            value=loss,
            batch_size=images.size(0),
            on_step=False,
            on_epoch=True
        )
        self.log(
            name='train_batch_loss',
            value=loss,
            batch_size=images.size(0),
            on_step=True,
            on_epoch=False
        )

        # Update metrics
        predictions = outputs.softmax(dim=-1).argmax(dim=-1)
        self.accuracy.update(predictions, labels)
        self.confusion_matrix.update(predictions, labels)
        self.per_class_accuracy.update(predictions, labels)
        self.f1_score.update(predictions, labels)

        return loss
    
    def _log_confusion_matrix(self, stage):
        # Compute confusion matrix
        conf_matrix = self.confusion_matrix.compute()

        # Get class names if label_encoder is available
        class_names = (self.label_encoder.classes_ if self.label_encoder 
                      else [f'Class {i}' for i in range(len(conf_matrix))])
        
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix.cpu().numpy(),
            annot=True,
            fmt='g',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{stage.capitalize()} Confusion Matrix')
        
        # Log the figure to tensorboard
        self.logger.experiment.add_figure(
            f'{stage}_confusion_matrix',
            fig,
            self.current_epoch
        )
        plt.close()
    
    def _log_per_class_metrics(self, stage):
        # Compute per-class metrics
        per_class_acc = self.per_class_accuracy.compute()
        f1_scores = self.f1_score.compute()

        # Get class names
        class_names = (self.label_encoder.classes_ if self.label_encoder 
                      else [f'Class {i}' for i in range(len(per_class_acc))])

        # Create a bar plot comparing accuracy and F1 score for each class
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(class_names))
        width = 0.35

        ax.bar(x - width/2, per_class_acc.cpu(), width, label='Accuracy')
        ax.bar(x + width/2, f1_scores.cpu(), width, label='F1 Score')



        ax.set_ylabel('Score')
        ax.set_title(f'{stage.capitalize()} Per-Class Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()

        plt.tight_layout()


        # Log the figure to tensorboard
        self.logger.experiment.add_figure(
            f'{stage}_per_class_metrics',
            fig,
            self.current_epoch
        )
        plt.close()

        # Log individual metrics
        for idx, class_name in enumerate(class_names):
            self.log(f'{stage}_acc_{class_name}', per_class_acc[idx])
            self.log(f'{stage}_f1_{class_name}', f1_scores[idx])

    def on_train_epoch_end(self):
        # Compute epoch metrics
        acc = self.accuracy.compute()

        # Log metrics
        self.log('train_acc', acc)






























        # Log confusion matrix and per-class metrics







































        self._log_confusion_matrix('train')
        self._log_per_class_metrics('train')

        # Reset metrics
        self.accuracy.reset()
        self.confusion_matrix.reset()
        self.per_class_accuracy.reset()
        self.f1_score.reset()

    def validation_step(self, batch, batch_idx):
        return self._evaluate(batch, stage='val')















    def test_step(self, batch, batch_idx):
        return self._evaluate(batch, stage='test')















    def on_test_epoch_end(self):
        # Compute epoch metrics
@@ -411,131 +469,125 @@
        self.f1_score.reset()

    def on_validation_epoch_end(self):
        # Compute epoch metrics
        acc = self.accuracy.compute()
        
        # Log metrics
        self.log('val_acc', acc)
        
        # Log confusion matrix and per-class metrics
        self._log_confusion_matrix('val')
        self._log_per_class_metrics('val')

        # Save model if validation accuracy improved
        if acc > getattr(self, 'best_val_acc', 0.0):
            self.best_val_acc = acc


            self.save_model()

        # Reset metrics
        self.accuracy.reset()
        self.confusion_matrix.reset()
        self.per_class_accuracy.reset()
        self.f1_score.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
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

    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'val_acc': self.best_val_acc,
        }, 'best_model.pth')
        print(f'Best model saved with validation accuracy: {self.best_val_acc:.2f}%')

def main():
    torch.manual_seed(42)

    # Hyperparameters
    batch_size = 64
    num_epochs = 200
    learning_rate = 1e-5

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
        version = "convnext_4_balanced_no_augmentation_v4",
        default_hp_metric=False
    )

    early_stop_callback = EarlyStopping(
            monitor='val_loss',  # quantity to monitor
            min_delta=0.00,            # minimum change to qualify as an improvement
            patience=30,               # number of epochs with no improvement after which training will be stopped
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
    model = MedicalTrainer(num_classes=4, learning_rate=learning_rate)
    model.set_label_encoder(label_encoder)
    trainer.fit(model, data_module)

    # Perform testing
    test_results = trainer.test(model, datamodule=data_module)


    # Final validation
    trainer.validate(model, data_module)

if __name__ == '__main__':
    main()
