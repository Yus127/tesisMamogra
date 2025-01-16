import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import lightning as L


"""
Linear probe, I added a extra layer to clasiffy the images into N categories 
"""
class CLIPLinearProbe(L.LightningModule):
    def __init__(
        self, 
        model, 
        class_descriptions: list, 
        learning_rate: float = 0.0001,
        weight_decay: float = 0.0, 
        dropout_rate: float = 0.2, 
        l2_lambda: float = 0.00
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
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", 
            num_classes=self.num_classes
        )

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
        
        # Update metrics
        _predictions = logits.softmax(dim=-1).argmax(dim=-1)
        self.accuracy.update(_predictions, labels)

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

        return loss

    def on_train_epoch_end(self):        
        # Compute epoch metrics
        acc = self.accuracy.compute()

        # Reset epoch loss counter
        self.train_epoch_loss = 0
        
        # Log metrics
        self.log('train_acc', acc)

    def validation_step(self, batch, batch_idx):
        self._evaluate(batch, stage='val')
    
    def on_validation_epoch_end(self):
        # Compute epoch metrics
        acc = self.accuracy.compute()
        # Log metrics
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        self._evaluate(batch, stage='test')
    
    def on_test_epoch_end(self):
        # Compute epoch metrics
        acc = self.accuracy.compute()
        # Log metrics
        self.log('test_acc', acc)

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