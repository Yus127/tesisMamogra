import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.text import BLEUScore
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
import torch.optim as optim

from pytorch_lightning.callbacks import EarlyStopping

from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
import math


"""
Text generation or image-captioning model using a pretrained CLIP model as a base 
for extracting image features, and adding a custom captioning head for generating 
text descriptions based on those features
"""

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

class LightningBiomedCLIP(pl.LightningModule):
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
                #print(f"\nTrue text: {true_texts[i]}")
                print(f"Generated text: {pred_texts[i]}")
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

class CLIPLinearProbe(pl.LightningModule):
    def __init__(
        self,
        model,
        class_descriptions,
        tokenizer, preprocess
        
    ):
        super().__init__()
        self.learning_rate = 0.0001
        self.num_classes = len(class_descriptions)
        self.class_descriptions = class_descriptions
        self.tokenizer = tokenizer

        
        # Load CLIP model
        self.clip_model = model
        self.preprocess = preprocess
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Get CLIP image embedding dimension
        with torch.no_grad():
            dummy_image = torch.zeros(1, 3, 224, 224, device=self.device)
            image_features = self.clip_model.encode_image(dummy_image)
            self.feature_dim = image_features.shape[1]
            
            # Encode class descriptions
            if class_descriptions:
                text_tokens = self.tokenizer([l for l in self.class_descriptions], context_length=17).to(self.device)
                self.class_text_features = self.clip_model.encode_text(text_tokens)
                self.class_text_features = F.normalize(self.class_text_features, dim=-1)
                self.class_text_features = self.class_text_features.to(self.device)
        
        
        # Linear probe layer
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)
        self.classifier = self.classifier.to(self.device)

        # Initialize weights manually
        nn.init.xavier_uniform_(self.classifier.weight, gain=1.0)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)
        """
        if self.classifier.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.classifier.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.classifier.bias, -bound, bound)
        """
        
        # Store text features for zero-shot prediction
        if class_descriptions:
            self.register_text_features()

    def _ensure_on_device(self, tensor, device=None):
        """Helper method to ensure tensor is on the correct device"""
        if device is None:
            device = self.device
        if tensor.device != device:
            tensor = tensor.to(device)
        return tensor

    def training_step(self, batch, batch_idx):
        images, text_tokens = batch['image'], batch['text']
        text_tokens = text_tokens.to(self.device)
        
        print("In training step - text_tokens shape:", text_tokens.shape)
        # Convert text tokens to class indices
        labels = self.get_class_index(text_tokens)
        print("After get_class_index - labels shape:", labels.shape)
        
        logits = self(images)
        print("Before cross_entropy - logits shape:", logits.shape)
        print("Before cross_entropy - labels:", labels)
        
        loss = F.cross_entropy(logits, labels)
        
        # Log training metrics
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss

    def register_text_features(self):
        with torch.no_grad():
            text_tokens =  self.tokenizer([l for l in self.class_descriptions], context_length=17).to(self.device)
            self.text_features = self.clip_model.encode_text(text_tokens)
            self.text_features = F.normalize(self.text_features, dim=-1)
    
    def get_class_index(self, text_tokens):
        with torch.no_grad():
            print("Text tokens shape:", text_tokens.shape)
            # Encode the input text
            text_features = self.clip_model.encode_text(text_tokens)
            print("Text features shape:", text_features.shape)
            text_features = F.normalize(text_features, dim=-1)
            
            # Ensure both tensors are on the same device
            if text_features.device != self.class_text_features.device:
                text_features = text_features.to(self.class_text_features.device)
            
            # Compare with class descriptions
            similarity = text_features @ self.class_text_features.t()
            print("Similarity shape:", similarity.shape)
            indices = similarity.argmax(dim=-1)
            print("Indices shape:", indices.shape)
            print("Indices content:", indices)
            return indices

    def forward(self, x):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(x)
            image_features = F.normalize(image_features, dim=-1)
        return self.classifier(image_features) *5.0

    
    def validation_step(self, batch, batch_idx):
        images, text_tokens = batch['image'], batch['text']
        
        # Ensure inputs are on the correct device
        images = self._ensure_on_device(images)
        text_tokens = self._ensure_on_device(text_tokens)
            
        print("In validation step - text_tokens shape:", text_tokens.shape)
        # Convert text tokens to class indices
        labels = self.get_class_index(text_tokens)
        labels = self._ensure_on_device(labels)
        print("After get_class_index - labels shape:", labels.shape)
        
        # Linear probe prediction
        logits = self(images)
        logits = self._ensure_on_device(logits)
        print("Before cross_entropy - logits shape:", logits.shape)
        print("Before cross_entropy - labels:", labels)
        predictions = logits.argmax(dim=-1)
        
        # Print predictions vs actual
        print("\nBatch", batch_idx, "Results:")
        print("=" * 50)
        for i, (pred, actual) in enumerate(zip(predictions, labels)):
            print(f"\nImage {i}:")
            print(f"Actual class   : {actual.item()} - {self.class_descriptions[actual.item()]}")
            print(f"Predicted class: {pred.item()} - {self.class_descriptions[pred.item()]}")
            
            # Print raw logits for diagnosis
            print("\nLogits distribution:")
            probs = F.softmax(logits[i], dim=0)
            for class_idx, (logit, prob) in enumerate(zip(logits[i], probs)):
                print(f"Class {class_idx:2d}: logit = {logit.item():8.3f}, prob = {prob.item():8.3f}")
            
        # Also print classifier weights statistics
        with torch.no_grad():
            weights = self.classifier.weight
            print("\nClassifier weights statistics:")
            print(f"Mean: {weights.mean().item():.3f}")
            print(f"Std : {weights.std().item():.3f}")
            print(f"Min : {weights.min().item():.3f}")
            print(f"Max : {weights.max().item():.3f}")

        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        
        # Zero-shot prediction
        if self.class_descriptions:
            zeroshot_logits = self.zeroshot_prediction(images)
            zeroshot_logits = self._ensure_on_device(zeroshot_logits)
            labels = self._ensure_on_device(labels)  # Ensure labels are still on correct device
            zeroshot_acc = (zeroshot_logits.argmax(dim=-1) == labels).float().mean()
            self.log('val_zeroshot_acc', zeroshot_acc, prog_bar=True)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return loss


    def zeroshot_prediction(self, x):
        with torch.no_grad():
            if x.device != self.device:
                x = x.to(self.device)
            image_features = self.clip_model.encode_image(x)
            image_features = F.normalize(image_features, dim=-1)
            
            # Ensure class features are on the correct device
            if self.class_text_features.device != self.device:
                self.class_text_features = self.class_text_features.to(self.device)
                
            similarity = image_features @ self.class_text_features.t()
            return similarity

    def configure_optimizers(self):
        return torch.optim.AdamW(self.classifier.parameters(), lr=self.learning_rate)
