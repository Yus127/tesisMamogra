import warnings
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import clip
from transformers import CLIPProcessor, CLIPModel

# Suppress warnings about TypedStorage
warnings.filterwarnings('ignore', message='TypedStorage is deprecated')

import os
import json
import torch
from PIL import Image
import numpy as np

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

def collate_fn(batch):
    """Custom collate function to handle batching of images and texts"""
    images, texts = zip(*batch)
    
    # Stack images
    images = torch.stack(images)
    
    # Pad texts to the same length (should already be padded, but just in case)
    texts = torch.stack(texts)
    
    return images, texts

def convert_models_to_fp32(model):
    """Convert model to float32 precision"""
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

def evaluate_predictions(model, dataloader, device, num_examples=5):
    """
    Show actual vs predicted text-image pairs
    Args:
        model: trained CLIP model
        dataloader: DataLoader instance
        device: torch device
        num_examples: number of examples to show
    Returns:
        dict: evaluation results
    """
    model.eval()
    
    with torch.no_grad():
        # Get a batch
        batch = next(iter(dataloader))
        if not batch:
            print("No data in dataloader")
            return None
        
        images, texts = batch
        images = images.to(device)
        texts = texts.to(device)
        
        # Get model predictions
        outputs = model(pixel_values=images, input_ids=texts)
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text
        
        # Get predictions
        image_preds = logits_per_image.argmax(dim=-1)
        text_preds = logits_per_text.argmax(dim=-1)
        
        # Get confidence scores
        image_scores = torch.softmax(logits_per_image, dim=-1)
        text_scores = torch.softmax(logits_per_text, dim=-1)
        
        return {
            'images': images.cpu()[:num_examples],
            'texts': texts.cpu()[:num_examples],
            'image_preds': image_preds.cpu()[:num_examples],
            'text_preds': text_preds.cpu()[:num_examples],
            'image_scores': image_scores.cpu()[:num_examples],
            'text_scores': text_scores.cpu()[:num_examples],
            'logits_per_image': logits_per_image.cpu()[:num_examples],
            'logits_per_text': logits_per_text.cpu()[:num_examples]
        }



def calculate_metrics(logits_per_image, logits_per_text):
    """
    Calculate retrieval metrics for CLIP
    Args:
        logits_per_image: Image-text similarity scores (batch_size, batch_size)
        logits_per_text: Text-image similarity scores (batch_size, batch_size)
    Returns:
        dict: Dictionary containing accuracy metrics
    """
    batch_size = logits_per_image.size(0)
    ground_truth = torch.arange(batch_size, device=logits_per_image.device)
    
    # Image to text retrieval accuracy
    image_pred = logits_per_image.argmax(dim=-1)
    i2t_accuracy = (image_pred == ground_truth).float().mean().item()
    
    # Text to image retrieval accuracy
    text_pred = logits_per_text.argmax(dim=-1)
    t2i_accuracy = (text_pred == ground_truth).float().mean().item()
    
    # Calculate top-5 accuracy
    _, top5_image_pred = logits_per_image.topk(5, dim=-1)
    i2t_top5 = (top5_image_pred == ground_truth.unsqueeze(-1)).any(dim=-1).float().mean().item()
    
    _, top5_text_pred = logits_per_text.topk(5, dim=-1)
    t2i_top5 = (top5_text_pred == ground_truth.unsqueeze(-1)).any(dim=-1).float().mean().item()
    
    return {
        'i2t_accuracy': i2t_accuracy,
        't2i_accuracy': t2i_accuracy,
        'i2t_top5': i2t_top5,
        't2i_top5': t2i_top5,
        'mean_accuracy': (i2t_accuracy + t2i_accuracy) / 2,
        'mean_top5': (i2t_top5 + t2i_top5) / 2
    }

def print_predictions(predictions, dataset, num_examples=5):
    """
    Print prediction results in a readable format
    Args:
        predictions: dict containing evaluation results
        dataset: dataset containing the samples
        num_examples: number of examples to show
    """
    if predictions is None:
        print("No predictions to show")
        return
        
    for i in range(min(num_examples, len(predictions['image_preds']))):
        print(f"\nExample {i+1}:")
        print("=" * 50)
        
        # Get actual text for this example
        actual_text = dataset.samples[i]['report']
        print(f"Actual Text: {actual_text}")
        
        # Get predicted text (based on image)
        pred_idx = predictions['image_preds'][i].item()
        pred_text = dataset.samples[pred_idx]['report']
        confidence = predictions['image_scores'][i][pred_idx].item()
        print(f"Predicted Text (confidence: {confidence:.3f}): {pred_text}")
        
        # Show top-3 text predictions
        logits = predictions['logits_per_image'][i]
        top_3_scores, top_3_indices = logits.topk(min(3, len(logits)))
        print("\nTop 3 Text Predictions:")
        for score, idx in zip(top_3_scores, top_3_indices):
            text = dataset.samples[idx]['report']
            print(f"Score: {score:.3f} - {text[:100]}...")

def calculate_loss(logits_per_image, logits_per_text, ground_truth):
    # Temperature scaling
    temperature = 0.07
    logits_per_image = logits_per_image / temperature
    logits_per_text = logits_per_text / temperature
    
    # Calculate InfoNCE loss
    loss_img = nn.CrossEntropyLoss()(logits_per_image, ground_truth)
    loss_txt = nn.CrossEntropyLoss()(logits_per_text, ground_truth)
    
    # Add contrastive loss component
    similarity = nn.CosineSimilarity()(logits_per_image, logits_per_text)
    contrastive_loss = (1 - similarity).mean()
    
    return (loss_img + loss_txt) / 2 + 0.1 * contrastive_loss

def train_clip_model(train_data_path, val_data_path, num_epochs=30, batch_size=32, learning_rate=5e-5):
    """
    Train CLIP model on mammogram images and reports
    Args:
        train_data_path (str): Path to training data JSON
        val_data_path (str): Path to validation data JSON
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimization
    Returns:
        model: Trained CLIP model
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model and datasets
    model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            ignore_mismatched_sizes=True  # Add this parameter
        ).to(device)
    
    
  

    train_dataset = ImageReportDataset(train_data_path)
    val_dataset = ImageReportDataset(val_data_path)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4 if device.type == 'cuda' else 0,
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if device.type == 'cuda' else 0,
        collate_fn=collate_fn
    )
    
    # Setup optimizer and loss functions
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.01
    )
     # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs,
        eta_min=learning_rate/100
    )
   
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')

    # Freeze base layers initially
    for param in model.vision_model.parameters():
        param.requires_grad = False
    for param in model.text_model.parameters():
        param.requires_grad = False
        
    # Train only top layers for first few epochs
    print("Training top layers only...")
    #for epoch in range(5):
    #    train_epoch(...)
    
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        train_metrics = []
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch_idx, (images, texts) in enumerate(train_pbar):
            optimizer.zero_grad()
            
            # Move batch to device
            images = images.to(device)
            texts = texts.to(device)
            
            # Forward pass
            outputs = model(pixel_values=images, input_ids=texts)
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text
            
            # Compute loss
            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
            loss = calculate_loss(logits_per_image, logits_per_text, ground_truth)
            #loss = (loss_img(logits_per_image, ground_truth) + 
            #       loss_txt(logits_per_text, ground_truth)) / 2
            
            # Backward pass
            loss.backward()
            
            # Optimizer step with precision handling
            if device.type == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
            
            batch_metrics = calculate_metrics(logits_per_image, logits_per_text)
            train_metrics.append(batch_metrics)
            
            total_train_loss += loss.item()
            train_pbar.set_postfix({
                'train_loss': f"{loss.item():.4f}",
                'i2t_acc': f"{batch_metrics['i2t_accuracy']:.3f}",
                't2i_acc': f"{batch_metrics['t2i_accuracy']:.3f}"
            })
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        val_metrics = []
        val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        
        with torch.no_grad():
            for batch_idx, (images, texts) in enumerate(val_pbar):
                images = images.to(device)
                texts = texts.to(device)
                
                outputs = model(pixel_values=images, input_ids=texts)
                logits_per_image = outputs.logits_per_image
                logits_per_text = outputs.logits_per_text
                
                ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
                val_loss = (loss_img(logits_per_image, ground_truth) + 
                           loss_txt(logits_per_text, ground_truth)) / 2
                
                batch_metrics = calculate_metrics(logits_per_image, logits_per_text)
                val_metrics.append(batch_metrics)
                
                total_val_loss += val_loss.item()
                val_pbar.set_postfix({
                    'val_loss': f"{val_loss.item():.4f}",
                    'i2t_acc': f"{batch_metrics['i2t_accuracy']:.3f}",
                    't2i_acc': f"{batch_metrics['t2i_accuracy']:.3f}"
                })
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        
        # Calculate epoch-level metrics
        epoch_metrics = {
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_i2t_acc': sum(m['i2t_accuracy'] for m in train_metrics) / len(train_metrics),
            'train_t2i_acc': sum(m['t2i_accuracy'] for m in train_metrics) / len(train_metrics),
            'train_i2t_top5': sum(m['i2t_top5'] for m in train_metrics) / len(train_metrics),
            'train_t2i_top5': sum(m['t2i_top5'] for m in train_metrics) / len(train_metrics),
            'val_i2t_acc': sum(m['i2t_accuracy'] for m in val_metrics) / len(val_metrics),
            'val_t2i_acc': sum(m['t2i_accuracy'] for m in val_metrics) / len(val_metrics),
            'val_i2t_top5': sum(m['i2t_top5'] for m in val_metrics) / len(val_metrics),
            'val_t2i_top5': sum(m['t2i_top5'] for m in val_metrics) / len(val_metrics),
        }
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Training Metrics:")
        print(f"  Loss: {epoch_metrics['train_loss']:.4f}")
        print(f"  Image->Text Accuracy: {epoch_metrics['train_i2t_acc']:.4f}")
        print(f"  Text->Image Accuracy: {epoch_metrics['train_t2i_acc']:.4f}")
        print(f"  Image->Text Top-5: {epoch_metrics['train_i2t_top5']:.4f}")
        print(f"  Text->Image Top-5: {epoch_metrics['train_t2i_top5']:.4f}")
        print(f"\nValidation Metrics:")
        print(f"  Loss: {epoch_metrics['val_loss']:.4f}")
        print(f"  Image->Text Accuracy: {epoch_metrics['val_i2t_acc']:.4f}")
        print(f"  Text->Image Accuracy: {epoch_metrics['val_t2i_acc']:.4f}")
        print(f"  Image->Text Top-5: {epoch_metrics['val_i2t_top5']:.4f}")
        print(f"  Text->Image Top-5: {epoch_metrics['val_t2i_top5']:.4f}")
        
        # Save checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'metrics': epoch_metrics
            }, checkpoint_path)
        
            print(f"\nSaved best model checkpoint with validation loss: {avg_val_loss:.4f}")
        
        # Save regular epoch checkpoint
        checkpoint_path = f'checkpoint_epoch_{epoch+1}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'metrics': epoch_metrics
        }, checkpoint_path)
    return model  # Return the trained model


def evaluate_model(model, dataset, device, num_examples=5):
    """
    Evaluate model predictions on a dataset
    Args:
        model: trained CLIP model
        dataset: dataset to evaluate on
        device: device to run evaluation on
        num_examples: number of examples to show
    Returns:
        dict: evaluation results
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    model = model.to(device)
    model.eval()
    
    dataloader = DataLoader(
        dataset,
        batch_size=num_examples,
        shuffle=True,
        num_workers=4 if device.type == 'cuda' else 0,
        collate_fn=collate_fn
    )
    
    try:
        predictions = evaluate_predictions(model, dataloader, device, num_examples)
        if predictions is not None:
            print_predictions(predictions, dataset, num_examples)
        return predictions
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    train_data_path ="/Users/YusMolina/Documents/tesis/biomedCLIP/data/datosMex/images/train.json"

    val_data_path ="/Users/YusMolina/Documents/tesis/biomedCLIP/data/datosMex/images/test.json"
    
    # Create datasets for evaluation
    train_dataset = ImageReportDataset(train_data_path)
    val_dataset = ImageReportDataset(val_data_path)
    

    import matplotlib.pyplot as plt

    # Create dataset

    # Retrieve a sample
    image, text_tensor = train_dataset[0]

    # Convert image tensor to numpy for visualization
    image_np = image.permute(1, 2, 0).numpy()

    # Display image
    plt.imshow(image_np)
    plt.title("Sample Image")
    plt.show()

    # Print corresponding text tensor
    print("Text Tensor:", text_tensor)

     # Train the model
    model = train_clip_model(
        train_data_path,
        val_data_path,
        num_epochs=30,
        batch_size=32,
        learning_rate=5e-5
    )

    # Evaluate on both training and validation sets
    print("\nEvaluating on Training Set:")
    train_results = evaluate_model(model, train_dataset, device="cuda" if torch.cuda.is_available() else "cpu")
    
    print("\nEvaluating on Validation Set:")
    val_results = evaluate_model(model, val_dataset, device="cuda" if torch.cuda.is_available() else "cpu")