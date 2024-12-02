import lightning as L
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import nrrd
import json
from PIL import Image


"""Expected JSON Structure
jsonCopy{
    "sample_id": {
        "image_paths": ["path1.jpg", "path2.jpg", ...],
        "mask_paths": ["path1.nrrd", "path2.nrrd", ...],
        "report": "medical report text"
    }
}
"""
class ComplexMedicalDataset(Dataset):
    def __init__(self, data_dir: str, processor, tokenizer):
        self.data_dir = data_dir
        self.processor = processor
        self.tokenizer = tokenizer
        
        with open(data_dir, 'r') as f:
            self.data = json.load(f)
        #print(self.data)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_dir = data_dir.rsplit('/', 1)[0]


    def __len__(self):
        return len(self.data)
    """
    def process_mask(self, mask_path):
        try:
            # Read NRRD file
            mask_data, _ = nrrd.read(os.path.join(self.data_dir, mask_path))
            
            # Ensure mask is in correct format and resize
            mask = torch.from_numpy(mask_data).float()
            
            # Resize mask to match image dimensions
            resize_transform = transforms.Resize((224, 224))
            
            # Handle different mask formats
            if mask.dim() == 2:  # Single channel mask
                mask = mask.unsqueeze(0)  # Add channel dimension
                mask = resize_transform(mask.unsqueeze(0)).squeeze(0)
                mask = mask.repeat(4, 1, 1)  # Expand to 4 channels
            elif mask.dim() == 3 and mask.size(0) != 4:  # Multi-channel mask but not 4 channels
                if mask.size(0) == 1:
                    mask = mask.repeat(4, 1, 1)
                else:
                    # Take first 4 channels or pad with zeros
                    temp_mask = torch.zeros((4, mask.size(1), mask.size(2)))
                    temp_mask[:min(4, mask.size(0))] = mask[:min(4, mask.size(0))]
                    mask = temp_mask
                mask = resize_transform(mask.unsqueeze(0)).squeeze(0)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            mask = mask.to(device)
            
            return mask

        except Exception as e:
            print(f"Error processing mask {mask_path}: {str(e)}")
            return torch.zeros((4, 224, 224))
    """
    
    def process_image(self, image_path):
        try:
            img = Image.open(os.path.join(self.data_dir, image_path)).convert('RGB')


            transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])
            processed_img = transform(img)
            if processed_img.dim() == 3:
                processed_img = processed_img.unsqueeze(0)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Use processor's image processing
            
            processed_img = processed_img.to(device)  
     
            return processed_img

        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return torch.zeros((1, 3, 224, 224))


    def __getitem__(self, idx: int) -> dict:
        #print(self.data)
        item = self.data[idx]
     
        a = list(item.keys())[0]

        # Process images
        images = []
        #masks = []
        #for img_path, mask_path in zip(item[a]['image_paths'], item[a]['mask_paths']):
        for img_path in item[a]['image_paths']:
            processed_img = self.process_image(img_path)
            #processed_mask = self.process_mask(mask_path)
            images.append(processed_img)
            #masks.append(processed_mask)


        # Stack images
        images = torch.stack(images)  # [N, 3, 224, 224]
        images = images.squeeze(1) 
        #masks = torch.stack(masks)

        #print("final images size")
        #print(images.size())

        # Process text using BiomedCLIP's text processor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        text = self.tokenizer(item[a]['report']).to(device)
        
        return {
            #"image": masks,
            "image": images,
            "text": text
        }



    @staticmethod
    def collate_fn(batch):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # Extract and process images
        all_images = []
        all_texts = []
        
        for item in batch:
            # Handle each image in the batch
            for img in item['image']:  
                all_images.append(img)
            all_texts.append(item['text'])
        
        # Stack all images
        images = torch.stack(all_images).to(device)  
        texts = torch.stack(all_texts)
        texts=texts.squeeze(1)
        texts.to(device)
        
        #print(f"Final batch image shape: {images.shape}")
        #print(f"Final batch text shape: {texts.shape}")
        
        return {
            'image': images,
            'text': texts
        }



class MyDatamodule(L.LightningDataModule):
    def __init__(self, data_dir:str, processor, tokenizer, batch_size:int = 32, num_workers:int = 4):
        super(MyDatamodule).__init__()
        # Dataset info
        self.data_dir = data_dir
        self.processor = processor
        self.tokenizer = tokenizer
        # Dataloaders info
        self.batch_size = batch_size
        self.num_workers = num_workers


    def setup(self, stage=None):
        '''
        Executes on every GPU. Setup the dataset for training, validation and testing.
        '''
        if stage == 'fit' or stage is None:
            self.train_dataset = ComplexMedicalDataset(self.data_dir, self.processor, self.tokenizer)
        if stage == 'test' or stage is None:
            self.test_dataset = ComplexMedicalDataset(self.data_dir, self.processor, self.tokenizer)
        

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=ComplexMedicalDataset.collate_fn
        )
    
    #def val_dataloader(self):
        '''
        # TODO: Implement
        return DataLoader(
            <val_dataset>,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=ComplexMedicalDataset.collate_fn
        )
        '''
        #return None
    
    #def test_dataloader(self):
        '''
        # TODO: Implement if needed
        return DataLoader(
            <test_dataset>,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=ComplexMedicalDataset.collate_fn
        )
        '''
        #return None
    
    """
    TODO REVISAR EL MODELO DE BIOMEDCLIP PARA VER SI RECIBE 4 DIMENSIONES O SOLO 3, Y DE ALLÍ ADECUAR EL CODIGO DEL DATASET O DEL MODELO PARA PODE PROCESAR LA MÁSCARA CON LA IMAGEN"""

    """"batches que entrene la gpu, 3090 """