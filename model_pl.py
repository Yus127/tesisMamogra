import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

"""
Linear probe, I added a extra layer to classify the images into N categories 
"""
class CLIPLinearProbe(L.LightningModule):
    def __init__(
        self, 
        model, 
        class_descriptions: list, 
        learning_rate: float,
        weight_decay: float, 
        dropout_rate: float, 
        l2_lambda: float
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        # Model initialization
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Get CLIP image embedding dimension
        self.feature_dim = ( # It is needed to know the model architecture
            self.model
            .get_submodule(target='visual')
            .get_submodule(target='head')
            .get_submodule(target='proj')
            .out_features
        )

        # Define an index for each class description
        self.class_text = class_descriptions
        self.num_classes = len(self.class_text)

        # Classifier initialization
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)
        nn.init.xavier_uniform_(self.classifier.weight, gain=1.4)
        nn.init.zeros_(self.classifier.bias)

        # Hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda

        # Metrics
        # Training metrics
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.train_confusion_matrix = torchmetrics.classification.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)
        self.train_per_class_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.num_classes, average=None)
        self.train_f1_score = torchmetrics.classification.F1Score(task="multiclass", num_classes=self.num_classes, average=None)
        
        # Validation metrics
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_confusion_matrix = torchmetrics.classification.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)
        self.val_per_class_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.num_classes, average=None)
        self.val_f1_score = torchmetrics.classification.F1Score(task="multiclass", num_classes=self.num_classes, average=None)
        
        # Test metrics
        self.test_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_confusion_matrix = torchmetrics.classification.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)
        self.test_per_class_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.num_classes, average=None)
        self.test_f1_score = torchmetrics.classification.F1Score(task="multiclass", num_classes=self.num_classes, average=None)

        self.metrics_updated = False

    def _common_step(self, batch, batch_idx):
        # Get batch
        image, text = batch['image'], batch['text']
        
        # Get label index
        labels = (torch.Tensor(len(text))
                .type(torch.LongTensor)
                .to(self.device)
        )
        for idx, t in enumerate(text):
            if t not in self.class_text:
                raise ValueError(
                    f"Class description '{t}' not found in target classes."
                )
            labels[idx] = torch.tensor(self.class_text.index(t))
        
        return image, labels

    def _evaluate(self, batch, stage=None):
        image, labels = self._common_step(batch, None)
        
        # Compute logits
        logits = self(image)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        # Update metrics based on stage
        _predictions = logits.softmax(dim=-1).argmax(dim=-1)
        if stage == 'val':
            self.val_accuracy.update(_predictions, labels)
            self.val_confusion_matrix.update(_predictions, labels)
            self.val_per_class_accuracy.update(_predictions, labels)
            self.val_f1_score.update(_predictions, labels)
        elif stage == 'test':
            self.test_accuracy.update(_predictions, labels)
            self.test_confusion_matrix.update(_predictions, labels)
            self.test_per_class_accuracy.update(_predictions, labels)
            self.test_f1_score.update(_predictions, labels)

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
        
        # Update metrics
        _predictions = logits.softmax(dim=-1).argmax(dim=-1)
        self.train_accuracy.update(_predictions, labels)
        self.train_confusion_matrix.update(_predictions, labels)
        self.train_per_class_accuracy.update(_predictions, labels)
        self.train_f1_score.update(_predictions, labels)
        self.metrics_updated = True

         # Log the accuracy for the current batch
        batch_acc = ((_predictions == labels).float().mean())

        # Log training metrics
        self.log(
            name='train_loss', 
            value=loss,  
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )
        self.log(
            name='train_acc_step', 
            value=batch_acc,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True
        )

        return loss

    def _log_confusion_matrix(self, stage):
        if not self.metrics_updated:
            return
        if stage == 'train':
            conf_matrix = self.train_confusion_matrix.compute()
        elif stage == 'val':
            conf_matrix = self.val_confusion_matrix.compute()
        elif stage == 'test':
            conf_matrix = self.test_confusion_matrix.compute()
        else:
            return
        
        # Convert class indices to class names for better interpretability
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
        
        # Log the figure to tensorboard
        self.logger.experiment.add_figure(
            f'{stage}_confusion_matrix',
            fig,
            self.current_epoch
        )
        plt.close()
        
    def _log_per_class_metrics(self, stage):
        if not self.metrics_updated:
            return

        # Get appropriate metrics based on stage
        if stage == 'train':
            per_class_acc = self.train_per_class_accuracy.compute()
            f1_scores = self.train_f1_score.compute()
        elif stage == 'val':
            per_class_acc = self.val_per_class_accuracy.compute()
            f1_scores = self.val_f1_score.compute()
        elif stage == 'test':
            per_class_acc = self.test_per_class_accuracy.compute()
            f1_scores = self.test_f1_score.compute()
        else:
            return
        
        # Create a bar plot comparing accuracy and F1 score for each class
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
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Log the figure to tensorboard
        self.logger.experiment.add_figure(
            f'{stage}_per_class_metrics',
            fig,
            self.current_epoch
        )
        plt.close()
        
        # Log individual metrics with sync_dist=True
        for idx, class_name in enumerate(self.class_text):
            self.log(
                f'{stage}_acc_{class_name}', 
                per_class_acc[idx],
                sync_dist=True
            )
            self.log(
                f'{stage}_f1_{class_name}', 
                f1_scores[idx],
                sync_dist=True
            )

    def on_train_epoch_end(self):
        if not self.metrics_updated:
            return

        # Compute epoch metrics
        acc = self.train_accuracy.compute()
    
        # Log metrics
        self.log('train_acc_epoch', acc, sync_dist=True, prog_bar=True)
        
        # Log confusion matrix and per-class metrics
        self._log_confusion_matrix('train')
        self._log_per_class_metrics('train')
        
        # Reset train metrics
        self.train_accuracy.reset()
        self.train_confusion_matrix.reset()
        self.train_per_class_accuracy.reset()
        self.train_f1_score.reset()
        self.metrics_updated = False

    def validation_step(self, batch, batch_idx):
        self._evaluate(batch, stage='val')
    
    def on_validation_epoch_end(self):
        if not self.metrics_updated:
            return
            
        # Compute epoch metrics
        acc = self.val_accuracy.compute()
        
        # Log metrics
        self.log('val_acc', acc, sync_dist=True, prog_bar=True)
        
        # Log confusion matrix and per-class metrics
        self._log_confusion_matrix('val')
        self._log_per_class_metrics('val')
        
        # Reset validation metrics
        self.val_accuracy.reset()
        self.val_confusion_matrix.reset()
        self.val_per_class_accuracy.reset()
        self.val_f1_score.reset()
        self.metrics_updated = False
    
    def test_step(self, batch, batch_idx):
        self._evaluate(batch, stage='test')
    
    def on_test_epoch_end(self):
        if not self.metrics_updated:
            return

        # Compute epoch metrics
        acc = self.test_accuracy.compute()
        
        # Log metrics
        self.log('test_acc', acc)
        
        # Log confusion matrix and per-class metrics
        self._log_confusion_matrix('test')
        self._log_per_class_metrics('test')
        
        # Reset test metrics
        self.test_accuracy.reset()
        self.test_confusion_matrix.reset()
        self.test_per_class_accuracy.reset()
        self.test_f1_score.reset()
        self.metrics_updated = False

    def forward(self, x):
        with torch.no_grad():
            image_features = self.model.encode_image(x)
        return self.classifier(image_features)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.classifier.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=0.2,
            patience=30,
            verbose=True,
            min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

