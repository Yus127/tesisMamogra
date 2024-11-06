
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.text import BLEUScore
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F

class CaptioningHead(nn.Module):
    def __init__(self, clip_hidden_size: int = 512, vocab_size: int = 28895, hidden_size: int = 512):
        super().__init__()
        self.dense = nn.Linear(clip_hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features):
        x = self.dropout(self.dense(features))
        x = self.out_proj(x)
        return x

class LightningBiomedCLIP(pl.LightningModule):
    def __init__(
        self,
        model,
        tokenizer,
        clip_hidden_size: int = 512,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_epochs: int = 10,
        hidden_size: int = 512,
        vocab_size: int = 28895
    ):
        super(LightningBiomedCLIP, self).__init__()
        self.save_hyperparameters(ignore=['model', 'tokenizer'])
        
        self.model = model
        self.tokenizer = tokenizer
        
        # Freeze CLIP model
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.caption_head = CaptioningHead(
            clip_hidden_size=clip_hidden_size,
            vocab_size=vocab_size,
            hidden_size=hidden_size
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is padding token
    
    def forward(self, images, texts=None):
        # Get image features from CLIP
        image_features, _, _ = self.model(images, texts)
        
        # Get predictions from captioning head
        logits = self.caption_head(image_features)  # [batch_size, vocab_size]
        
        return logits
    
    def training_step(self, batch, batch_idx):
        images = batch['image']
        texts = batch['text']  # Already tokenized [batch_size, seq_len]
        
        # Move to device if needed
        texts = texts.to(self.device)
        
        # Get logits
        logits = self(images)  # [batch_size, vocab_size]

        #print(logits)
        
        # Calculate loss on first token prediction
        loss = self.criterion(logits, texts[:, 0])  # Using first token as target
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        
        # Calculate accuracy
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=-1)  # [batch_size]
            print(predictions)
            accuracy = (predictions == texts[:, 0]).float().mean()
            self.log('train_acc', accuracy, prog_bar=True)
            
            if batch_idx % 100 == 0:
                print(f"\nBatch {batch_idx}")
                print(f"Loss: {loss.item():.4f}")
                print(f"Accuracy: {accuracy.item():.4f}")
                print(f"\nShapes:")
                print(f"Images: {images.shape}")
                print(f"Text tokens: {texts.shape}")
                print(f"Logits: {logits.shape}")
                print(f"Predictions: {predictions.shape}")
                
                # Print example predictions
                print("\nTraining Examples:")
                for i in range(min(2, len(predictions))):
                    print(f"True token: {texts[i, 0].item()}")
                    print(f"Predicted token: {predictions[i].item()}")
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images = batch['image']
        texts = batch['text']  # Already tokenized
        
        # Move to device if needed
        texts = texts.to(self.device)
        
        # Get logits
        logits = self(images)  # [batch_size, vocab_size]
        
        # Calculate loss
        loss = self.criterion(logits, texts[:, 0])
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        
        # Calculate accuracy
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == texts[:, 0]).float().mean()
            self.log('val_acc', accuracy, prog_bar=True)
            
            if batch_idx == 0:
                print("\nValidation Examples:")
                for i in range(min(3, len(predictions))):
                    print(f"True token: {texts[i, 0].item()}")
                    print(f"Predicted token: {predictions[i].item()}")
        
        return loss
    
    def generate(self, images):
        """
        Generate token predictions from images
        """
        with torch.no_grad():
            logits = self(images)  # [batch_size, vocab_size]
            predictions = torch.argmax(logits, dim=-1)  # [batch_size]
            
        return predictions
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.caption_head.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        total_steps = self.trainer.estimated_stepping_batches
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=total_steps
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }