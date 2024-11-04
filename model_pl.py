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

    def generate_caption(self, image_path: str) -> str:
        #Generate caption for a single image
        transform = Compose([
            Resize(224),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), 
                     (0.26862954, 0.26130258, 0.27577711))
        ])
        
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(self.device)
        
        self.eval()
        with torch.no_grad():
            logits = self(image)
            predicted_ids = torch.argmax(logits, dim=-1)
            caption = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        
        return caption
    """
