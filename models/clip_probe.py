"""
Linear probe classifier on top of a frozen BioMedCLIP encoder for breast
density classification. Only the final linear layer is trained.

Training Strategy:
    - Linear probing: BioMedCLIP encoder weights are frozen
    - AdamW optimizer with weight decay
    - ReduceLROnPlateau scheduler
    - Early stopping based on validation loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class CLIPLinearProbe(L.LightningModule):
    """
    Input Image → [Frozen BioMedCLIP Encoder] → Embeddings
    → [Trainable Linear Layer] → Class Logits
    """

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

        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False

        self.feature_dim = (
            self.model
            .get_submodule(target='visual')
            .get_submodule(target='head')
            .get_submodule(target='proj')
            .out_features
        )

        self.class_text = class_descriptions
        self.num_classes = len(self.class_text)

        self.classifier = nn.Linear(self.feature_dim, self.num_classes)
        nn.init.xavier_uniform_(self.classifier.weight, gain=1.4)
        nn.init.zeros_(self.classifier.bias)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda

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
        with torch.no_grad():
            image_features = self.model.encode_image(x)
        return self.classifier(image_features)

    def _common_step(self, batch, batch_idx):
        image, text = batch['image'], batch['text']
        labels = torch.tensor(
            [self.class_text.index(t) for t in text],
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
        if not self.metrics_updated:
            return
        conf_matrix = getattr(self, f'{stage}_confusion_matrix').compute()
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix.cpu().numpy(),
            annot=True, fmt='g',
            xticklabels=self.class_text,
            yticklabels=self.class_text
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{stage.capitalize()} Confusion Matrix')
        self.logger.experiment.add_figure(f'{stage}_confusion_matrix', fig, self.current_epoch)
        plt.close()

    def _log_per_class_metrics(self, stage):
        if not self.metrics_updated:
            return
        per_class_acc = getattr(self, f'{stage}_per_class_accuracy').compute()
        f1_scores = getattr(self, f'{stage}_f1_score').compute()

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
        self.logger.experiment.add_figure(f'{stage}_per_class_metrics', fig, self.current_epoch)
        plt.close()

        for idx, class_name in enumerate(self.class_text):
            self.log(f'{stage}_acc_{class_name}', per_class_acc[idx], sync_dist=True)
            self.log(f'{stage}_f1_{class_name}', f1_scores[idx], sync_dist=True)

    def training_step(self, batch, batch_idx):
        image, labels = self._common_step(batch, batch_idx)
        logits = self(image)

        loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        if self.l2_lambda > 0.0:
            l2_norm = sum(torch.sum(p ** 2) for p in self.classifier.parameters())
            loss = loss + self.l2_lambda * l2_norm

        predictions = logits.softmax(dim=-1).argmax(dim=-1)
        self._update_metrics('train', predictions, labels)
        self.metrics_updated = True

        batch_acc = (predictions == labels).float().mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_acc_step', batch_acc, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        if not self.metrics_updated:
            return
        acc = self.train_accuracy.compute()
        self.log('train_acc_epoch', acc, sync_dist=True, prog_bar=True)
        self._log_confusion_matrix('train')
        self._log_per_class_metrics('train')
        self._reset_metrics('train')
        self.metrics_updated = False

    def validation_step(self, batch, batch_idx):
        image, labels = self._common_step(batch, None)
        logits = self(image)
        loss = F.cross_entropy(logits, labels)
        predictions = logits.softmax(dim=-1).argmax(dim=-1)
        self._update_metrics('val', predictions, labels)
        self.metrics_updated = True
        self.log('val_loss', loss, batch_size=image.size(0))

    def on_validation_epoch_end(self):
        if not self.metrics_updated:
            return
        acc = self.val_accuracy.compute()
        self.log('val_acc', acc, sync_dist=True, prog_bar=True)
        self._log_confusion_matrix('val')
        self._log_per_class_metrics('val')
        self._reset_metrics('val')
        self.metrics_updated = False

    def test_step(self, batch, batch_idx):
        image, labels = self._common_step(batch, None)
        logits = self(image)
        loss = F.cross_entropy(logits, labels)
        predictions = logits.softmax(dim=-1).argmax(dim=-1)
        self._update_metrics('test', predictions, labels)
        self.metrics_updated = True
        self.log('test_loss', loss, batch_size=image.size(0))

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
        optimizer = torch.optim.AdamW(
            params=self.classifier.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode='min', factor=0.2, patience=30,
            verbose=True, min_lr=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
