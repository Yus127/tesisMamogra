import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import lightning as L

#from torchmetrics.text import BLEUScore
#from transformers import get_linear_schedule_with_warmup

""" TODO: Clean this classes
Text generation or image-captioning model using a pretrained CLIP model as a base 
for extracting image features, and adding a custom captioning head for generating 
text descriptions based on those features

class CaptioningHead(nn.Module):
    def __init__(self, clip_hidden_size: int, vocab_size: int, hidden_size: int ):
        super().__init__()
        self.dense = nn.Linear(clip_hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features):
        x = self.dense(features)
        #x = self.dropout(self.dense(features))
        x = self.out_proj(x)
        return x

class LightningBiomedCLIP(L.LightningModule):
    def __init__(
        self,
        model,
        tokenizer,
        clip_hidden_size: int,
        learning_rate: float,
        weight_decay: float,
        warmup_steps: int,
        hidden_size: int,
        vocab_size: int,
        max_length: int,  
        bos_token_id: int, 
        eos_token_id: int,  
        pad_token_id: int   
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
                #print(f"\nTrue text: {true_texts[i]}")
                print(f"Generated text: {pred_texts[i]}")
                print(f"BLEU score: {bleu_score:.4f}")
        
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        print("validation step ")
        images = batch['image']
        texts = batch['text']  # Already tokenized [batch_size, seq_len]
        
        # Move to device if needed
        texts = texts.to(self.device)
        
        # Get logits
        logits = self(images)  # [batch_size, vocab_size]
        ce_loss = self.criterion(logits, texts[:, 0])
        
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
                #print(f"\nTrue text: {true_texts[i]}")
                print(f"Generated text: {pred_texts[i]}")
                print(f"BLEU score: {bleu_score:.4f}")
        
        
        return loss
    
    def test_step(self, batch, batch_idx):
        print("Testing step")
        
        images = batch['image']
        texts = batch['text']  # Already tokenized [batch_size, seq_len]
        
        # Move to device if needed
        texts = texts.to(self.device)
        
        # Get logits
        logits = self(images)  # [batch_size, vocab_size]
        ce_loss = self.criterion(logits, texts[:, 0])
        
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
                #print(f"\nTrue text: {true_texts[i]}")
                print(f"Generated text: {pred_texts[i]}")
                print(f"BLEU score: {bleu_score:.4f}")
        
        
        return loss

    
    def generate(self, images, max_length=None, temperature=1.0, top_k=50):
        
        # Generate complete token sequences using top-k sampling
        
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
                logits = self.caption_head(image_features)  
                
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
"""


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
        self.epoch_loss = 0 

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
        self.epoch_loss += loss.item()
        
        # Log training metrics
        self.log('train_batch_loss', loss, on_step=True)
        
        # Update metrics
        _predictions = logits.softmax(dim=-1).argmax(dim=-1)
        self.accuracy.update(_predictions, labels)

        return loss

    def _epoch_metrics(self, epoch_loss):
        # Compute epoch metrics
        acc = self.accuracy.compute()
        loss = epoch_loss / self.trainer.num_training_batches

        # Reset metrics
        self.epoch_loss = 0

        return acc, loss

    def on_train_epoch_end(self):
        # Compute epoch metrics
        acc, loss = self._epoch_metrics(self.epoch_loss)
        
        # Log metrics
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_acc', acc, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        image, labels = self._common_step(batch, batch_idx)
        
        # Compute logits
        with torch.no_grad():
            logits = self(image)

        # Compute loss
        loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        self.epoch_loss += loss.item()

        # Log validation metrics
        self.log('val_batch_loss', loss, on_step=True)

        # Update metrics
        _predictions = logits.softmax(dim=-1).argmax(dim=-1)
        self.accuracy.update(_predictions, labels)

        return loss
    
    def on_validation_epoch_end(self):
        # Compute epoch metrics
        acc, loss = self._epoch_metrics(self.epoch_loss)

        # Log metrics
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)

    def test_step(self, batch, batch_idx):
        image, labels = self._common_step(batch, batch_idx)
        
        # Compute logits
        with torch.no_grad():
            logits = self(image)

        # Compute loss
        loss = F.cross_entropy(logits, labels)
        self.epoch_loss += loss.item()

        # Update metrics
        _predictions = logits.softmax(dim=-1).argmax(dim=-1)
        self.accuracy.update(_predictions, labels)

        return loss
    
    def on_test_epoch_end(self):
        # Compute epoch metrics
        acc, loss = self._epoch_metrics(self.epoch_loss)
        
        # Log metrics
        self.log('test_epoch_loss', loss, on_epoch=True)
        self.log('test_epoch_acc', acc, on_epoch=True)

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