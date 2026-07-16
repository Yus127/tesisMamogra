"""ConvNeXt-Base fine-tuning for breast density classification."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision.models import convnext_base, ConvNeXt_Base_Weights


class ConvNeXtClassifier(L.LightningModule):

    def __init__(self, class_descriptions: list, learning_rate: float = 0.0001):
        super().__init__()
        self.save_hyperparameters()

        self.class_descriptions = class_descriptions
        self.num_classes = len(class_descriptions)
        self.learning_rate = learning_rate

        self.model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        self.model.classifier[2] = nn.Linear(1024, self.num_classes)

        for stage in ('train', 'val', 'test'):
            setattr(self, f'{stage}_accuracy', torchmetrics.classification.Accuracy(
                task="multiclass", num_classes=self.num_classes))
            setattr(self, f'{stage}_confusion_matrix', torchmetrics.classification.ConfusionMatrix(
                task="multiclass", num_classes=self.num_classes))
            setattr(self, f'{stage}_per_class_accuracy', torchmetrics.classification.Accuracy(
                task="multiclass", num_classes=self.num_classes, average=None))
            setattr(self, f'{stage}_f1_score', torchmetrics.classification.F1Score(
                task="multiclass", num_classes=self.num_classes, average=None))

        self.metrics_updated = False

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        image, text = batch['image'], batch['text']
        labels = torch.tensor(
            [self.class_descriptions.index(t) for t in text],
            dtype=torch.long, device=self.device
        )
        return image, labels

    def _update_metrics(self, stage, predictions, labels):
        getattr(self, f'{stage}_accuracy').update(predictions, labels)
        getattr(self, f'{stage}_confusion_matrix').update(predictions, labels)
        getattr(self, f'{stage}_per_class_accuracy').update(predictions, labels)
        getattr(self, f'{stage}_f1_score').update(predictions, labels)

    def _reset_metrics(self, stage):
        getattr(self, f'{stage}_accuracy').reset()
        getattr(self, f'{stage}_confusion_matrix').reset()
        getattr(self, f'{stage}_per_class_accuracy').reset()
        getattr(self, f'{stage}_f1_score').reset()

    def _log_confusion_matrix(self, stage):
        conf_matrix = getattr(self, f'{stage}_confusion_matrix').compute()
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix.cpu().numpy(),
            annot=True, fmt='g', cmap='Blues',
            xticklabels=self.class_descriptions,
            yticklabels=self.class_descriptions
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{stage.capitalize()} Confusion Matrix')
        plt.tight_layout()
        self.logger.experiment.add_figure(f'{stage}_confusion_matrix', fig, self.current_epoch)
        plt.close()

    def _log_per_class_metrics(self, stage):
        per_class_acc = getattr(self, f'{stage}_per_class_accuracy').compute()
        f1_scores = getattr(self, f'{stage}_f1_score').compute()

        for idx, class_name in enumerate(self.class_descriptions):
            self.log(f'{stage}_acc_{class_name}', per_class_acc[idx])
            self.log(f'{stage}_f1_{class_name}', f1_scores[idx])

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(self.class_descriptions))
        width = 0.35
        ax.bar(x - width/2, per_class_acc.cpu(), width, label='Accuracy')
        ax.bar(x + width/2, f1_scores.cpu(), width, label='F1 Score')
        ax.set_ylabel('Score')
        ax.set_title(f'{stage.capitalize()} Per-Class Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_descriptions, rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        self.logger.experiment.add_figure(f'{stage}_per_class_metrics', fig, self.current_epoch)
        plt.close()

    def training_step(self, batch, batch_idx):
        image, labels = self._common_step(batch, batch_idx)
        logits = self(image)
        loss = F.cross_entropy(logits, labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_batch_loss', loss, on_step=True, on_epoch=False)
        predictions = logits.softmax(dim=-1).argmax(dim=-1)
        self._update_metrics('train', predictions, labels)
        self.metrics_updated = True
        return loss

    def on_train_epoch_end(self):
        if not self.metrics_updated:
            return
        acc = self.train_accuracy.compute()
        self.log('train_acc', acc)
        self._log_confusion_matrix('train')
        self._log_per_class_metrics('train')
        self._reset_metrics('train')
        self.metrics_updated = False

    def validation_step(self, batch, batch_idx):
        image, labels = self._common_step(batch, None)
        logits = self(image)
        loss = F.cross_entropy(logits, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        predictions = logits.softmax(dim=-1).argmax(dim=-1)
        self._update_metrics('val', predictions, labels)
        self.metrics_updated = True

    def on_validation_epoch_end(self):
        if not self.metrics_updated:
            return
        acc = self.val_accuracy.compute()
        self.log('val_acc', acc)
        self._log_confusion_matrix('val')
        self._log_per_class_metrics('val')
        self._reset_metrics('val')
        self.metrics_updated = False

    def test_step(self, batch, batch_idx):
        image, labels = self._common_step(batch, None)
        logits = self(image)
        loss = F.cross_entropy(logits, labels)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        predictions = logits.softmax(dim=-1).argmax(dim=-1)
        self._update_metrics('test', predictions, labels)
        self.metrics_updated = True

    def on_test_epoch_end(self):
        if not self.metrics_updated:
            return
        acc = self.test_accuracy.compute()
        self.log('test_acc', acc)
        self._log_confusion_matrix('test')
        self._log_per_class_metrics('test')
        self._reset_metrics('test')
        self.metrics_updated = False

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50, verbose=True
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
