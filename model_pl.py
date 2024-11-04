"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.text import BLEUScore
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F

class CaptioningHead(nn.Module):
    def __init__(self, clip_hidden_size: int = 512, vocab_size: int= 28895, hidden_size: int = 512):
        super().__init__()
        self.dense = nn.Linear(clip_hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, texts):
       
        x = self.dropout(self.dense(texts))
 
        x = self.out_proj(x)
      
        return x

class LightningBiomedCLIP(pl.LightningModule):
    def __init__(
        self,
        model,
        tokenizer,
        clip_hidden_size: int = 512,  # Changed to match VisionTransformer output
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
            
        # Initialize captioning head
        self.caption_head = CaptioningHead(
            clip_hidden_size=clip_hidden_size,
            vocab_size=vocab_size,
            hidden_size=hidden_size
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        self.bleu = BLEUScore()
    
    def forward(self, images, texts=None):
        # Get image features from CLIP
        image_features, text_features, logit_scale = self.model(images, texts)
        
        # Get prediction from captioning head
        predictions = self.caption_head(image_features)
        
        return predictions, text_features, logit_scale
    
    def training_step(self, batch, batch_idx):
        images, texts = batch['image'], batch['text']
        
        # Forward pass through CLIP and captioning head
        predictions, text_features, logit_scale = self(images, texts) #forward
        
        # Calculate similarity scores
        #logits = logit_scale * predictions @ text_features.t()
        logits = (logit_scale * predictions @ text_features.t()).detach().softmax(dim=0)
    
        print(logits)

        # For contrastive loss, labels are the diagonal elements (matching pairs)
        labels = torch.arange(predictions.size(0), device=self.device)

        print(labels)
        
        # Calculate symmetric contrastive loss
        loss_i2t = self.criterion(logits, labels)
        loss_t2i = self.criterion(logits.t(), labels)
        loss = (loss_i2t + loss_t2i) / 2.0
        
        # Calculate metrics
        with torch.no_grad():
            # Calculate accuracy
            pred_i2t = torch.argmax(logits, dim=1)
            pred_t2i = torch.argmax(logits, dim=0)
            acc_i2t = (pred_i2t == labels).float().mean()
            acc_t2i = (pred_t2i == labels).float().mean()
            
            # Log metrics
            self.log('train_loss', loss, prog_bar=True)
            self.log('train_acc_i2t', acc_i2t, prog_bar=True)
            self.log('train_acc_t2i', acc_t2i, prog_bar=True)
            
            # Log similarity matrix statistics
            self.log('mean_similarity', logits.mean())
            self.log('max_similarity', logits.max())
            self.log('min_similarity', logits.min())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, texts = batch['image'], batch['text']
        
        # Forward pass
        predictions, text_features, logit_scale = self(images, texts) #forward
        
        # Calculate similarity scores
        logits = logit_scale * predictions @ text_features.t()
        
        # Calculate contrastive loss
        labels = torch.arange(predictions.size(0), device=self.device)
        loss_i2t = self.criterion(logits, labels)
        loss_t2i = self.criterion(logits.t(), labels)
        loss = (loss_i2t + loss_t2i) / 2.0
        
        pred_indices = torch.argmax(logits, dim=1)

        # Calculate metrics
        pred_i2t = torch.argmax(logits, dim=1)
        pred_t2i = torch.argmax(logits, dim=0)
        acc_i2t = (pred_i2t == labels).float().mean()
        acc_t2i = (pred_t2i == labels).float().mean()
        
        # Log validation metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc_i2t', acc_i2t, prog_bar=True)
        self.log('val_acc_t2i', acc_t2i, prog_bar=True)

        if batch_idx == 0:
            print("\nValidation Examples:")
            print("-------------------")
            for i in range(min(3, len(texts))):
                true_text = texts[i]
                pred_text = texts[pred_indices[i]]
                print(f"\nExample {i+1}:")
                print(f"True text: {true_text}")
                print(f"Predicted text: {pred_text}")
                print(f"Similarity score: {logits[i, pred_indices[i]]:.4f}")
            
        return loss
    
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
    
    def predict_step(self, batch, batch_idx):
        images = batch['image']
        predictions, _, _ = self(images) #forward
        return predictions
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.text import BLEUScore
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F

class CaptioningHead(nn.Module):
    def __init__(self, clip_hidden_size: int = 512, output_size: int = 512, hidden_size: int = 512):
        super().__init__()
        self.dense = nn.Linear(clip_hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, output_size)  # Changed to output CLIP embedding size
        
    def forward(self, features):
        x = self.dropout(self.dense(features))
        x = self.out_proj(x)
        # Normalize output to match CLIP's feature space
        x = F.normalize(x, p=2, dim=1)
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
        hidden_size: int = 512
    ):
        super(LightningBiomedCLIP, self).__init__()
        self.save_hyperparameters(ignore=['model', 'tokenizer'])
        
        self.model = model
        self.tokenizer = tokenizer
        
        # Freeze CLIP model
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Initialize captioning head to output CLIP embedding size
        self.caption_head = CaptioningHead(
            clip_hidden_size=clip_hidden_size,
            output_size=clip_hidden_size,  # Match CLIP's embedding size
            hidden_size=hidden_size
        )
        
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, images, texts=None):
        # Get features from CLIP model
        image_features, text_features, logit_scale = self.model(images, texts)
        
        # Get prediction from captioning head (will be in same space as text_features)
        predictions = self.caption_head(image_features)
        
        return predictions, text_features, logit_scale
    
    def training_step(self, batch, batch_idx):
        images, texts = batch['image'], batch['text']
        
        # Forward pass through CLIP and captioning head
        predictions, text_features, logit_scale = self(images, texts)
        print(predictions.shape)
        print(predictions)
        # Calculate similarity scores with predictions and text features
        # Now both predictions and text_features are [batch_size, 512]
        logits = logit_scale * torch.matmul(predictions, text_features.t())
        
        # For contrastive loss, labels are the diagonal elements (matching pairs)
        labels = torch.arange(predictions.size(0), device=self.device)
        print(labels)
        
        # Calculate loss
        loss = self.criterion(logits, labels)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        
        # Calculate accuracy
        with torch.no_grad():
            pred_indices = torch.argmax(logits, dim=1)
            accuracy = (pred_indices == labels).float().mean()
            self.log('train_acc', accuracy, prog_bar=True)
            
            if batch_idx % 100 == 0:
                print(f"\nBatch {batch_idx}")
                print(f"Loss: {loss.item():.4f}")
                print(f"Accuracy: {accuracy.item():.4f}")
                #print(f"Predictions shape: {predictions.shape}")  # Should be [7, 512]
                #print(f"Text features shape: {text_features.shape}")  # Should be [7, 512]
                #print(f"Logits shape: {logits.shape}")  # Should be [7, 7]
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, texts = batch['image'], batch['text']
        
        # Forward pass
        predictions, text_features, logit_scale = self(images, texts)
        
        # Calculate similarity scores
        logits = logit_scale * torch.matmul(predictions, text_features.t())
        
        # Calculate loss
        labels = torch.arange(predictions.size(0), device=self.device)
        loss = self.criterion(logits, labels)
        
        # Log validation metrics
        self.log('val_loss', loss, prog_bar=True)
        
        # Calculate accuracy
        with torch.no_grad():
            pred_indices = torch.argmax(logits, dim=1)
            accuracy = (pred_indices == labels).float().mean()
            self.log('val_acc', accuracy, prog_bar=True)
            
            if batch_idx == 0:
                print("\nValidation Examples:")
                print("-------------------")
                for i in range(min(3, len(texts))):
                    print(f"\nExample {i+1}:")
                    print(f"True text: {texts[i]}")
                    print(f"Matched text: {texts[pred_indices[i]]}")
                    print(f"Similarity score: {logits[i, pred_indices[i]]:.4f}")
        
        return loss
    
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

    def predict_step(self, batch, batch_idx):
        images = batch['image']
        predictions, _, _ = self(images)
        return predictions