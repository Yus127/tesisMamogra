import lightning as L
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import json
from PIL import Image


"""Expected JSON Structure
jsonCopy{
    "sample_id": {
        "image_paths": ["path1.jpg", "path2.jpg", ...],
        "report": "medical report text"
    }
}
"""
class ComplexMedicalDataset(Dataset):
    def __init__(self, data_dir: str, processor, tokenizer):
        self.data_dir = data_dir
        self.processor = processor
        self.tokenizer = tokenizer
        
        with open(os.path.join(data_dir, 'dataset_info.json'), 'r') as f:
            self.data = json.load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def __len__(self):
        return len(self.data)
    
    
    def process_image(self, image_path):
        print("Preporcess-image")
        try:
            img = Image.open(os.path.join(self.data_dir, image_path)).convert('RGB')
            #print(img)
            

            transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            processed_img = transform(img)
            #print("the size of ")
            if processed_img.dim() == 3:
                processed_img = processed_img.unsqueeze(0)
            
            device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

            # Use processor's image processing
            
            processed_img = processed_img.to(device)  # Move to correct device
            #print(processed_img.size())
            # todo check if we dont include the encode_image here    
            #processed = self.processor.encode_image(processed_img)
            #print(processed.size())
            return processed_img

        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return torch.zeros((1, 3, 224, 224))


    def __getitem__(self, idx: int) -> dict:
        print("get item")
        item = self.data[idx]
     
        a = list(item.keys())[0]

        # Process images
        #print(item[a]['image_paths'])
        images = []
        for img_path in item[a]['image_paths']:
            processed_img = self.process_image(img_path)
            #print(processed_img.size())
            images.append(processed_img)

        # Stack images
        images = torch.stack(images)  # [N, 3, 224, 224]
        images = images.squeeze(1) 

        print("final images size")
        print(images.size())

        # Process text using BiomedCLIP's text processor
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        text = self.tokenizer(item[a]['report']).to(device)
        #print(text)
        
        return {
            "image": images,
            "text": text
        }



    @staticmethod
    def collate_fn(batch):
        print("collate")
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
        
        print(f"Final batch image shape: {images.shape}")
        print(f"Final batch text shape: {texts.shape}")
        
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
        TODO: Implement the validation and testing dataset if needed.
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
    
    def val_dataloader(self):
        '''
        # TODO: Implement if needed
        return DataLoader(
            <val_dataset>,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=ComplexMedicalDataset.collate_fn
        )
        '''
        return None
    
    def test_dataloader(self):
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
        return None