import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.text import BLEUScore
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_lightning.callbacks import EarlyStopping
import numpy as np

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

"""
Linear probe, I added a extra layer to clasiffy the images into 17 categories 
"""
class CLIPLinearProbe(pl.LightningModule):
    def __init__(
        self,
        model,
        class_descriptions,
        tokenizer, 
        preprocess, 
        data_augmentation
    ):
        super().__init__()
        
        # Store initialization parameters
        self.class_descriptions = class_descriptions
        self.tokenizer = tokenizer
        self.clip_model = model
        self.preprocess = preprocess
        self.data_augmentation = data_augmentation
        self.num_classes = len(class_descriptions)
        self.dropout_rate = 0.2
        self.weight_decay = 0.01
        self.learning_rate = 0.0001
        
        # Save hyperparameters for TensorBoard
        self.save_hyperparameters(ignore=['model'])
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Get CLIP image embedding dimension
        with torch.no_grad():
            dummy_image = torch.zeros(1, 3, 224, 224, device=self.device)
            image_features = self.clip_model.encode_image(dummy_image)
            self.feature_dim = image_features.shape[1]
            
            # Encode class descriptions
            text_tokens = self.tokenizer([l for l in self.class_descriptions], context_length=17).to(self.device)
            self.class_text_features = self.clip_model.encode_text(text_tokens)
            self.class_text_features = F.normalize(self.class_text_features, dim=-1)
            self.class_text_features = self.class_text_features.to(self.device)

            # Store similarities for later logging
            self.similarities = self.class_text_features @ self.class_text_features.t()

            # Print similarities for debugging
            print("\nClass Description Similarities:")
            for i in range(len(class_descriptions)):
                most_similar = torch.topk(self.similarities[i], 3)
                print(f"\nClass {i} most similar to:")
                for sim, idx in zip(most_similar.values, most_similar.indices):
                    if idx != i:
                        print(f"Class {idx}: {sim:.3f}")
        
        # Data augmentation
        self.train_transforms = T.Compose([
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(10),
            T.RandomAffine(degrees=(1,10), translate=(0.1, 0.1), scale=(0.9, 1.1))
        ])

        # Classifier initialization
        self.classifier = nn.Linear(self.feature_dim, self.num_classes).to(self.device)
        nn.init.xavier_uniform_(self.classifier.weight, gain=1.4)
        #nn.init.zeros_(self.classifier.bias)

    def on_fit_start(self):
        """Called when fit begins; logger is guaranteed to exist at this point."""
        if self.logger is not None:
            # Log similarity matrix
            fig = plt.figure(figsize=(10, 10))
            sns.heatmap(self.similarities.cpu().numpy(), annot=True, fmt='.2f')
            plt.title('Class Description Similarities')
            self.logger.experiment.add_figure('class_similarities', fig, 0)
            plt.close(fig)

            # Log model graph
            dummy_input = torch.zeros(1, self.feature_dim, device=self.device)
            self.logger.experiment.add_graph(self.classifier, dummy_input)

    def training_step(self, batch, batch_idx):
        images, text_tokens = batch['image'], batch['text']
        text_tokens = text_tokens.to(self.device)
        
        # Convert text tokens to class indices
        labels = self.get_class_index(text_tokens)
        #logits = self(images)
        logit_scale = 100.0  # You can adjust this temperature value as needed
        logits = (logit_scale * images @ text_tokens.t()).detach().softmax(dim=-1)
        sorted_indices = torch.argsort(logits, dim=-1, descending=True)

        # Compute loss with label smoothing
        loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        
        # L2 regularization
        l2_lambda = 0.01
        l2_norm = sum(p.pow(2.0).sum() for p in self.classifier.parameters())
        loss = loss + l2_lambda * l2_norm
        
        # Log training metrics
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        self.log('l2_norm', l2_norm, prog_bar=False)
        
        # Periodically log visuals if logger exists
        if self.logger is not None and batch_idx % 10 == 0:
            # Log images
            x = images[:8]
            grid = torchvision.utils.make_grid(x.view(-1,1,224,224))
            self.logger.experiment.add_image("training_samples", grid, self.global_step)
            
            # Log predictions
            predictions = logits.argmax(dim=-1)
            
            # Create prediction distribution plot
            pred_hist = torch.zeros(self.num_classes)
            for i in range(self.num_classes):
                pred_hist[i] = (predictions == i).float().mean()
            
            fig = plt.figure(figsize=(10, 5))
            plt.bar(range(self.num_classes), pred_hist.cpu().numpy())
            plt.title('Prediction Distribution')
            plt.xlabel('Class')
            plt.ylabel('Frequency')
            self.logger.experiment.add_figure('pred_distribution', fig, self.global_step)
            plt.close(fig)
            
            # Log confusion matrix
            confusion = torch.zeros(self.num_classes, self.num_classes)
            for pred, label in zip(predictions, labels):
                confusion[label][pred] += 1
            
            fig = plt.figure(figsize=(10, 10))
            sns.heatmap(confusion.cpu().numpy(), annot=True, fmt='g')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            self.logger.experiment.add_figure('confusion_matrix', fig, self.global_step)
            plt.close(fig)
            
            print(f"\nTraining Batch {batch_idx}")
            print(f"Loss: {loss.item():.4f}")
            print("Predictions:", predictions.tolist())
            print("Actual:", labels.tolist())
        
        return loss

    def validation_step(self, batch, batch_idx):
        #images, text_tokens = batch['image'], batch['text']
        image_features, text_features, logit_scale = self.clip_model(batch['image'], batch['text'])

        
        image_features = self._ensure_on_device(image_features)
        text_features = self._ensure_on_device(text_features)
        
        labels = self.get_class_index(text_features)
        labels = self._ensure_on_device(labels)
        
        n_tta = 5  # Test Time Augmentation
        """
        all_logits = []
        for _ in range(n_tta):
            logits = self(image_features)
            all_logits.append(logits)
        """
        logits = (100 * image_features @ text_features.t()).detach().softmax(dim=-1)

        #logits = torch.stack(all_logits).mean(0)
        logits = self._ensure_on_device(logits)
        predictions = logits.argmax(dim=-1)
        
        loss = F.cross_entropy(logits, labels)
        acc = (predictions == labels).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        # Log validation visuals if logger exists
        if self.logger is not None and batch_idx == 0:
            # Log confusion matrix
            confusion = torch.zeros(self.num_classes, self.num_classes)
            for pred, label in zip(predictions, labels):
                confusion[label][pred] += 1
            
            fig = plt.figure(figsize=(10, 10))
            sns.heatmap(confusion.cpu().numpy(), annot=True, fmt='g')
            plt.title('Validation Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            self.logger.experiment.add_figure('val_confusion_matrix', fig, self.global_step)
            plt.close(fig)
            
            # Log validation samples
            grid = torchvision.utils.make_grid(images[:8].view(-1,1,224,224))
            self.logger.experiment.add_image("validation_samples", grid, self.global_step)
        
        return loss

    def on_train_epoch_end(self):
        """Log histograms at the end of each training epoch."""
        if self.logger is not None:
            for name, param in self.classifier.named_parameters():
                self.logger.experiment.add_histogram(f'classifier/{name}', param, self.current_epoch)
                if param.grad is not None:
                    self.logger.experiment.add_histogram(f'classifier/{name}_grad', param.grad, self.current_epoch)

    # Existing methods remain the same
    def _ensure_on_device(self, tensor, device=None):
        if device is None:
            device = self.device
        if tensor.device != device:
            tensor = tensor.to(device)
        return tensor
    
    def get_class_index(self, text_tokens):
        with torch.no_grad():
            # Encode the input text
            text_features = self.clip_model.encode_text(text_tokens)

            text_features = F.normalize(text_features, dim=-1)
            
            text_features = self._ensure_on_device(text_features)

            # Compare with class descriptions
                        
            self.class_text_features = self._ensure_on_device(self.class_text_features)

            similarity = text_features @ self.class_text_features.t()
            indices = similarity.argmax(dim=-1)
            #logit_scale = 100.0  # You can adjust this temperature value as needed
            #logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
            #sorted_indices = torch.argsort(logits, dim=-1, descending=True)
            return indices


    def forward(self, x):
        with torch.no_grad():
            if self.training and self.data_augmentation:
                x = self.train_transforms(x)
            image_features = self.clip_model.encode_image(x)
            image_features = F.normalize(image_features, dim=-1)
        return self.classifier(image_features)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
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