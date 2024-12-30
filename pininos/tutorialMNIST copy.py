"""
this i dont think it will work im oretending to do the same i did before, to predict all the tokens,
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import multiprocessing
from torch.multiprocessing import freeze_support
from transformers import ConvNextModel, AutoTokenizer
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms

import os
import json
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ImageReportDataset(Dataset):
    def __init__(self, data_dir):
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        """
        Initialize dataset from a JSON file containing mammogram data
        Args:
            data_dir (str): Path to JSON file containing dataset information
        """
        # Add logging to check data loading
        self.base_path = os.path.dirname(data_dir)
        
        # Load and parse JSON data
        with open(data_dir, 'r') as f:
            data_list = json.load(f)
        print(f"Loaded {len(data_list)} samples from JSON")

        # Setup image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(
                degrees=10, 
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
        ])
        
        # Restructure data and add logging
        self.samples = []
        for item in data_list:
            for image_name, content in item.items():
                if 'image_paths' in content and 'report' in content:
                    report = content['report'].strip()  # Remove extra whitespace
                    if len(report) > 0:  # Only add if report is not empty
                        self.samples.append({
                            'image_name': image_name,
                            'image_path': content['image_paths'][0],
                            'report': report
                        })
        print(f"Processed {len(self.samples)} valid samples")
    
    def process_image(self, image_path):
        """
        Process a single image, handling both regular images and TIFF/TIF files
        """
        try:
            full_path = os.path.join(self.base_path, image_path)
            
            # Open the image
            img = Image.open(full_path)
            
            # Handle TIFF/TIF specific processing
            if img.format in ['TIFF', 'TIF']:
                # Convert to numpy array
                img_array = np.array(img)
                
                # Handle different bit depths
                if img_array.dtype != np.uint8:
                    # Normalize to 8-bit range
                    img_array = ((img_array - img_array.min()) * (255.0 / (img_array.max() - img_array.min()))).astype(np.uint8)
                
                # Handle different channel configurations
                if len(img_array.shape) == 2:  # Single channel
                    # Convert to RGB by duplicating the channel
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif len(img_array.shape) == 3:
                    if img_array.shape[2] == 1:  # Single channel in 3D
                        img_array = np.concatenate([img_array] * 3, axis=2)
                    elif img_array.shape[2] == 4:  # RGBA
                        img_array = img_array[:, :, :3]  # Take only RGB channels
                
                # Convert back to PIL
                img = Image.fromarray(img_array)
            else:
                # For non-TIF images, convert to RGB
                img = img.convert('RGB')
            
            # Apply the standard transformation pipeline
            processed_img = self.transform(img)
            return processed_img
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return torch.zeros((3, 224, 224))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a single item from the dataset"""
        sample = self.samples[idx]
        
        # Process image - will return tensor of shape [3, 224, 224]
        image = self.process_image(sample['image_path'])
            
        # Process report text using CLIP processor
        text_features = self.processor(
            text=sample['report'], 
            return_tensors="pt", 
            padding='max_length', 
            truncation=True,
            max_length=77  # CLIP's maximum sequence length
        )
        
        # Remove the batch dimension added by the processor
        text_tensor = text_features['input_ids'].squeeze(0)
        
        return image, text_tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import multiprocessing
from torch.multiprocessing import freeze_support
from transformers import ConvNextModel, AutoTokenizer

class ImageReportNN(nn.Module):
    def __init__(self, hidden_size=768, num_classes=49408, max_length=77):
        super().__init__()
        self.max_length = max_length
        
        # Keep existing layers...
        self.convnext = ConvNextModel.from_pretrained("facebook/convnext-base-224")
        for param in self.convnext.parameters():
            param.requires_grad = False
            
        self.image_projection = nn.Linear(1024, hidden_size)
        self.text_embedding = nn.Embedding(num_classes, hidden_size)
        
        # Modified fusion layer to include position information
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Add a transformer decoder layer for sequence generation
        self.decoder = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1
        )
        
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, images, text_tensor=None, generate=False):
        batch_size = images.shape[0]
        device = images.device
        
        # Process images
        image_output = self.convnext(images)
        image_features = image_output.pooler_output
        image_features = self.image_projection(image_features)
        
        if not generate:
            # Training mode - teacher forcing
            # Create position indices
            positions = torch.arange(text_tensor.shape[1], device=device)
            positions = positions.unsqueeze(0).expand(batch_size, -1)
            pos_embeddings = self.position_embedding(positions)
            
            # Get text embeddings
            text_embeds = self.text_embedding(text_tensor)
            text_features = text_embeds + pos_embeddings
            
            # Process through transformer decoder
            output = self.decoder(text_features, image_features.unsqueeze(1))
            
            # Get logits for each position
            logits = self.classifier(output)
            return logits
        else:
            # Generation mode
            generated = []
            current_token = torch.full((batch_size, 1), 49406, device=device)  # <|startoftext|> token
            
            for i in range(self.max_length):
                # Get position embedding for current position
                pos_embedding = self.position_embedding(torch.tensor([i], device=device))
                pos_embedding = pos_embedding.expand(batch_size, -1)
                
                # Get text embedding for current tokens
                text_embed = self.text_embedding(current_token[:, -1:])
                text_features = text_embed + pos_embedding.unsqueeze(1)
                
                # Process through transformer decoder
                output = self.decoder(text_features, image_features.unsqueeze(1))
                
                # Get next token prediction
                logits = self.classifier(output)
                next_token = torch.argmax(logits[:, -1, :], dim=-1)
                
                # Add to generated sequence
                generated.append(next_token)
                current_token = torch.cat([current_token, next_token.unsqueeze(1)], dim=1)
                
                # Stop if we predict end token
                if (next_token == 49407).all():  # <|endoftext|> token
                    break
            
            return torch.stack(generated, dim=1)   

def train_model(train_dataset, val_dataset=None, num_epochs=10):
    # Set number of workers based on system
    if torch.cuda.is_available():
        num_workers = min(4, multiprocessing.cpu_count())
    else:
        num_workers = 0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 16
    learning_rate = 1e-4
    
    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False
    )
    
    if val_dataset:
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False
        )
    
    # Initialize model
    model = ImageReportNN().to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Use tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (images, text_tensor) in enumerate(pbar):
            try:
                # Move data to device
                images = images.to(device)      # Shape: [batch_size, 3, 224, 224]
                text_tensor = text_tensor.to(device)  # Shape: [batch_size, seq_len]

                check_tensor_shapes({
                    'batch_images': images,
                    'batch_text': text_tensor
                })
                
                # Forward pass
                logits = model(images, text_tensor)  # Shape: [batch_size, vocab_size]

                check_tensor_shapes({
                    'logits_before_loss': logits,
                    'text_tensor_before_loss': text_tensor
                })
                # Reshape for loss computation
                B, S, V = logits.shape
                logits = logits.view(-1, V)
                targets = text_tensor.view(-1)
                

                # Make sure logits has the correct shape [batch_size, num_classes]
                if len(logits.shape) == 3:
                    logits = logits.squeeze(1)  # Remove the middle dimension if it exists
                    
                # For loss computation, we'll predict the first token of the sequence
                #targets = text_tensor[:, 0]  # Shape: [batch_size]

                loss = criterion(logits, targets)
                # Get predictions
                predictions = torch.argmax(logits, dim=1)
                
                # Print predictions and targets for the first few examples in batch
                if batch_idx % 10 == 0:  # Print every 10th batch
                    print("\nBatch", batch_idx)
                    print("Predictions vs Targets:")

                    # Generate full sequence
                    with torch.no_grad():
                        predicted_sequence = model(images[:3], generate=True)
                    
                    for i in range(min(3, predicted_sequence.shape[0])):
                        pred_text = tokenizer.decode(predicted_sequence[i])
                        target_text = tokenizer.decode(text_tensor[i])
                        print(f"Prediction: {pred_text}")
                        print(f"Target: {target_text}")
                        print("---")

                    

                check_tensor_shapes({
                    'logits_at_loss': logits,
                    'targets_at_loss': targets
                })

                # Compute loss using the first token
                loss = criterion(logits, targets)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'batch': f'{batch_idx}/{len(train_loader)}'
                })
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                print(f"Image shape: {images.shape}")
                print(f"Text tensor shape: {text_tensor.shape}")
                continue
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
        
        # Validation
        if val_dataset:
            validate_model(model, val_loader, criterion, device, tokenizer)
   
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        
    
    
    return model

def check_tensor_shapes(tensor_dict):
    """Helper function to debug tensor shapes"""
    for name, tensor in tensor_dict.items():
        if isinstance(tensor, torch.Tensor):
            print(f"{name} shape: {tensor.shape}")

def validate_model(model, val_loader, criterion, device, tokenizer):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    
    with torch.no_grad():
        for batch_idx, (images, text_tensor) in enumerate(val_loader):
            try:
                # Move data to device
                images = images.to(device)
                text_tensor = text_tensor.to(device)
                
                # Forward pass
                logits = model(images, text_tensor)
                
                # Get targets - use the first token like in training
                targets = text_tensor[:, 0]
                
                 # Get predictions
                predictions = torch.argmax(logits, dim=1)
                
                # Store predictions and targets
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Compute loss
                loss = criterion(logits, targets)
                total_loss += loss.item()

                # Compute loss
                loss = criterion(logits, targets)
                total_loss += loss.item()

                # Print every 5th batch in validation
                if batch_idx % 5 == 0:
                    print("\nValidation Batch", batch_idx)
                    print("Predictions vs Targets:")
                    for i in range(min(3, len(predictions))):
                        pred_token = predictions[i].item()
                        target_token = targets[i].item()
                        try:
                            pred_text = tokenizer.decode([pred_token])
                            target_text = tokenizer.decode([target_token])
                            print(f"Prediction: {pred_text} ({pred_token})")
                            print(f"Target: {target_text} ({target_token})")
                            print("---")
                        except Exception as e:
                            print(f"Error decoding tokens: {e}")
                
                
            except Exception as e:
                print(f"Error in validation: {str(e)}")
                print(f"Image shape: {images.shape}")
                print(f"Text tensor shape: {text_tensor.shape}")
                if 'logits' in locals():
                    print(f"Logits shape: {logits.shape}")
                continue
    
    avg_loss = total_loss / len(val_loader)
    print(f'\nValidation Loss: {avg_loss:.4f}')
    
    # Print some overall statistics
    predictions_array = np.array(all_predictions)
    targets_array = np.array(all_targets)
    accuracy = np.mean(predictions_array == targets_array)
    print(f'Validation Accuracy: {accuracy:.4f}')
    
    model.train()

def main():
    # Initialize multiprocessing support
    freeze_support()
    
    # Create your dataset
    dataset = ImageReportDataset(data_dir ="/Users/YusMolina/Documents/tesis/biomedCLIP/data/datosMex/images/train.json")
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Train the model
    model = train_model(train_dataset, val_dataset, num_epochs=10)
    
    return model

if __name__ == '__main__':
    main()