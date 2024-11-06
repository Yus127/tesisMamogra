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
        vocab_size: int = 28895,
        max_length: int = 64,  
        bos_token_id: int = 101, 
        eos_token_id: int = 102,  
        pad_token_id: int = 0,   
    ):
        super(LightningBiomedCLIP, self).__init__()
        self.save_hyperparameters(ignore=['model', 'tokenizer'])
        
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        
        # Freeze CLIP model
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.caption_head = CaptioningHead(
            clip_hidden_size=clip_hidden_size,
            vocab_size=vocab_size,
            hidden_size=hidden_size
        )
        self.bleu = BLEUScore()

        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)  # Assuming 0 is padding token
    
    def forward(self, images, texts=None):
        # Get image features from CLIP
        image_features, _, _ = self.model(images, texts)
        
        # Get predictions from captioning head
        logits = self.caption_head(image_features)  # [batch_size, vocab_size]
        
        return logits
    
    def decode_tokens(self, tokens):
        # Handles various input formats and cleans up special tokens
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy().tolist()
        
        # Clean up special tokens and padding
        if isinstance(tokens, list):
            try:
                eos_idx = tokens.index(self.eos_token_id)
                tokens = tokens[:eos_idx]
            except ValueError:
                pass
                
            if tokens and tokens[0] == self.bos_token_id:
                tokens = tokens[1:]
                
            tokens = [t for t in tokens if t != self.pad_token_id]
        
        # Try different decoding methods
        try:
            text = self.tokenizer.tokenizer.decode(tokens)
        except:
            try:
                text = self.tokenizer.tokenizer.tokenizer.decode(tokens)
            except:
                text = str(tokens)
                print("Error in token " + text)
                
        return text
    
    def training_step(self, batch, batch_idx):
        images = batch['image']
        texts = batch['text']  # Already tokenized [batch_size, seq_len]
        
        # Move to device if needed
        texts = texts.to(self.device)
        
        # Get logits
        logits = self(images)  # [batch_size, vocab_size]
        ce_loss = self.criterion(logits, texts[:, 0])
        #print(logits)
        
        # Generate complete sequences
        generated_seqs = self.generate(images)  # [batch_size, seq_len]

        true_texts = []
        pred_texts = []
        
        # Convert true and generated sequences to text
        for i in range(len(texts)):
            # Decode true sequence
            true_text = self.decode_tokens(texts[i])
            true_texts.append(true_text)
            
            # Decode predicted sequence
            pred_text = self.decode_tokens(generated_seqs[i])
            pred_texts.append(pred_text)



        bleu_score = self.bleu(pred_texts, [true_texts])

        

        # Combined loss (weighted sum)
        loss = ce_loss - 0.1 * bleu_score  # Negative because we want to maximize BLEU
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_bleu', bleu_score, prog_bar=True)
        self.log('train_ce_loss', ce_loss, prog_bar=True)
        
        if batch_idx % 100 == 0:
            print("\nTraining Examples:")
            for i in range(min(2, len(true_texts))):
                print(f"\nTrue text: {true_texts[i]}")
                print(f"Generated text: {pred_texts[i]}")
                print(f"BLEU score: {bleu_score:.4f}")
        
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        print("validation step ")
        images = batch['image']
        texts = batch['text']
        
        # Generate complete sequences
        generated_seqs = self.generate(images)
        
        # Convert sequences to text
        true_texts = []
        pred_texts = []
        
        for i in range(len(texts)):
            true_text = self.decode_tokens(texts[i])
            true_texts.append(true_text)
            
            pred_text = self.decode_tokens(generated_seqs[i])
            pred_texts.append(pred_text)
        
        # Calculate BLEU score
      

        bleu_score = self.bleu(pred_texts, [[text] for text in true_texts])
        
        # Calculate cross entropy loss
        logits = self(images)
        ce_loss = self.criterion(logits, texts[:, 0])
        
        # Combined loss
        loss = ce_loss - 0.1 * bleu_score
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_bleu', bleu_score, prog_bar=True)
        self.log('val_ce_loss', ce_loss, prog_bar=True)
        
        if batch_idx == 0:
            print("\nValidation Examples:")
            for i in range(min(3, len(true_texts))):
                #print(f"\nTrue text: {true_texts[i]}")
                #print(f"Generated text: {pred_texts[i]}")
                print(f"BLEU score: {bleu_score:.4f}")
        
        return loss
    
    def generate(self, images, max_length=None, temperature=1.0, top_k=50):
        """
        Generate complete token sequences using top-k sampling
        """
        if max_length is None:
            max_length = self.max_length
            
        batch_size = images.size(0)
        
        # Initialize sequences with BOS token
        sequences = torch.full(
            (batch_size, 1),
            self.bos_token_id,
            dtype=torch.long,
            device=self.device
        )
        
        # Cache image features to avoid recomputing
        with torch.no_grad():
            image_features, _, _ = self.model(images)
        
        # Generate tokens autoregressively
        for _ in range(max_length - 1):
            # Get logits from the caption head
            with torch.no_grad():
                logits = self.caption_head(image_features)  # [batch_size, vocab_size]
                
                # Apply temperature
                logits = logits / temperature
                
                # Apply top-k sampling
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next tokens
                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
            
            # Append new tokens to sequences
            sequences = torch.cat([sequences, next_tokens], dim=1)
            
            # Check if all sequences have generated EOS token
            if (next_tokens == self.eos_token_id).all():
                break
        
        return sequences
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.caption_head.parameters(),
            lr=self.learning_rate, #self.
            weight_decay=self.weight_decay ##self.hparams
        )
        
        total_steps = self.trainer.estimated_stepping_batches
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,#self.hparams
            num_training_steps=total_steps
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }