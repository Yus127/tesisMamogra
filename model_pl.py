import lightning as L
from torchmetrics.text import BLEUScore
import pytorch_lightning as pl
import torch
from torch import optim
#def my_custom_loss()
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

"""
class CaptioningHead(nn.Module):
    def __init__(self, clip_hidden_size: int, vocab_size: int, hidden_size: int = 512, max_seq_length: int = 256):
        super().__init__()
        self.dense = nn.Linear(clip_hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, vocab_size)
        self.max_seq_length = max_seq_length

        
    def forward(self, x):
        # x shape: [batch_size, clip_hidden_size]
        batch_size = x.size(0)
        
        # Project to hidden size
        x = self.dropout(self.dense(x))  # [batch_size, hidden_size]
        
        # Expand to sequence length
        x = x.unsqueeze(1).expand(-1, self.max_seq_length, -1)  # [batch_size, seq_length, hidden_size]
        
        # Project to vocabulary size
        x = self.out_proj(x)  # [batch_size, seq_length, vocab_size]
        
        return x




class LightningBiomedCLIP(pl.LightningModule):
    def __init__(self, model,tokenizer, clip_hidden_size: int = 224,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_epochs: int = 10,
        hidden_size: int = 224):
        super(LightningBiomedCLIP, self).__init__()
        self.model = model
        self.tokenizer= tokenizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        self.clip_hidden_size = clip_hidden_size
        self.hidden_size = hidden_size

        # Freeze the CLIP model
        for param in self.model.parameters():
            param.requires_grad = False
       

        self.caption_head = CaptioningHead(
            clip_hidden_size=224, #aqui 
            vocab_size=224,
            hidden_size=512,
            max_seq_length=256
        )
        #self.loss = loss
        self.bleu = BLEUScore()
        #self.optimizer = optimizer
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, x):
        # Get image features from CLIP
        with torch.no_grad():
            image_features = self.model.get_image_features(x)  # [batch_size, clip_hidden_size]
        
        # Generate captions
        caption_logits = self.caption_head(image_features)  # [batch_size, seq_length, vocab_size]
        return caption_logits
        
   
   
    
    def training_step(self, batch, batch_idx):
        
        images, captions = batch['image'], batch['text']
        print(images)
        batch_size = images.size(0)
        sequence_length = captions.size(1)

        # Forward pass
        image_features, text_features, logits = self.model(images, captions)
        #logits = self(images)
        # Reshape for loss calculation
        print(logits)
        print("Logits")
        logits = logits.view(-1, logits.size(-1))  # [batch_size * seq_length, vocab_size]
        captions = captions.view(-1)  

        
        # Calculate loss
        loss = self.criterion(
            logits, captions
        )
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        return loss

    

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.caption_head.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        total_steps = self.trainer.estimated_stepping_batches
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch['image'], val_batch['text']
  """
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.text import BLEUScore
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F

class CaptioningHead(nn.Module):
    def __init__(self, clip_hidden_size: int, vocab_size: int, hidden_size: int = 512):
        super().__init__()
        self.dense = nn.Linear(clip_hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        x = self.dropout(self.dense(x))
        x = self.out_proj(x)
        return x

class LightningBiomedCLIP(pl.LightningModule):
    def __init__(
        self,
        model,
        tokenizer,
        clip_hidden_size: int = 768,  # Changed to match VisionTransformer output
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_epochs: int = 10,
        hidden_size: int = 512,
        vocab_size: int = 30522
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
        predictions, text_features, logit_scale = self(images, texts)
        
        # Calculate similarity scores
        logits = logit_scale * predictions @ text_features.t()
        
        # For contrastive loss, labels are the diagonal elements (matching pairs)
        labels = torch.arange(predictions.size(0), device=self.device)
        
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
        predictions, text_features, logit_scale = self(images, texts)
        
        # Calculate similarity scores
        logits = logit_scale * predictions @ text_features.t()
        
        # Calculate contrastive loss
        labels = torch.arange(predictions.size(0), device=self.device)
        loss_i2t = self.criterion(logits, labels)
        loss_t2i = self.criterion(logits.t(), labels)
        loss = (loss_i2t + loss_t2i) / 2.0
        
        # Calculate metrics
        pred_i2t = torch.argmax(logits, dim=1)
        pred_t2i = torch.argmax(logits, dim=0)
        acc_i2t = (pred_i2t == labels).float().mean()
        acc_t2i = (pred_t2i == labels).float().mean()
        
        # Log validation metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc_i2t', acc_i2t, prog_bar=True)
        self.log('val_acc_t2i', acc_t2i, prog_bar=True)
        
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