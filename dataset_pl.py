import os
import json
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import lightning as L
from dotenv import load_dotenv

load_dotenv()

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
    def __init__(self, data_dir:str, tokenizer, train:bool=True, transform=None):
        super(ComplexMedicalDataset, self).__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.transform = transform

        if train:
            with open(os.path.join(data_dir, "train.json"), 'r') as f:
                self.data = json.load(f)
        else:
            with open(os.path.join(data_dir, "test.json"), 'r') as f:
                self.data = json.load(f)

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx: int) -> dict:
        item_dict = self.data[idx]
        unique_key = next(iter(item_dict)) # Getting the image path here is odd due to the data structure
        
        # Load image
        image_path = item_dict[unique_key]["image_paths"][0]
        image = cv2.imread(os.path.join(self.data_dir, image_path))

        if self.transform:
            image = self.transform(image)

        # Load text
        text = self.tokenizer(item_dict[unique_key]['report'])
        
        return {"image": image, "text": text}


class MyDatamodule(L.LightningDataModule):
    def __init__(self, data_dir:str, tokenizer, transforms:dict, batch_size:int=32, num_workers:int=1):
        super(MyDatamodule, self).__init__()

        # Dataset info
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.transforms = transforms
        # Dataloaders info
        self.batch_size = batch_size
        self.num_workers = num_workers


    def setup(self, stage=None):
        '''
        Executes on every GPU. Setup the dataset for training, validation and testing.
        '''
        # Load all training data
        training_data = ComplexMedicalDataset(
            data_dir=self.data_dir,
            tokenizer=self.tokenizer,
            train=True,
            transform=self.transforms['train']
        )
        
        # Split training data into training and validation
        self.train_dataset, self.validation_dataset = torch.utils.data.random_split(training_data, [0.8, 0.2])

        # Load test data
        self.test_dataset = ComplexMedicalDataset(data_dir=self.data_dir, tokenizer=self.tokenizer, train=False, transform=self.transforms['test'])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


def _test_ComplexMedicalDataset():

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])

    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    myMedicalDataset = ComplexMedicalDataset(data_dir=os.getenv("DATA_DIR"), tokenizer=tokenizer, transform=train_transform)

    first_item = myMedicalDataset.__getitem__(0)

    # Visualize the first image
    print(f"First item shape: {first_item['image'].shape}")
    plt.imshow(first_item['image'].squeeze().permute(1, 2, 0))
    plt.show()

    # Visualize the first text
    print(first_item['text'])


def _test_MyDatamodule():
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])

    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    myMedicalDataModule = MyDatamodule(
        data_dir=os.getenv("DATA_DIR"),
        tokenizer=tokenizer,
        transforms={'train': train_transform, 'test': train_transform},
        batch_size=2,
        num_workers=1)

    myMedicalDataModule.setup()

    first_train_item = myMedicalDataModule.train_dataset.__getitem__(0)
    first_val_item = myMedicalDataModule.validation_dataset.__getitem__(0)
    first_test_item = myMedicalDataModule.test_dataset.__getitem__(0)

    # Visualize the first image of each dataset
    print(f"First train item shape: {first_train_item['image'].shape}")
    plt.imshow(first_train_item["image"].squeeze().permute(1, 2, 0))
    plt.show()

    print(f"First val item shape: {first_val_item['image'].shape}")
    plt.imshow(first_val_item["image"].squeeze().permute(1, 2, 0))
    plt.show()

    print(f"First test item shape: {first_test_item['image'].shape}")
    plt.imshow(first_test_item["image"].squeeze().permute(1, 2, 0))
    plt.show()

    # Visualize the first text of each dataset
    print(f"First train text {first_train_item['text']}")
    print(f"First val text {first_val_item['text']}")
    print(f"First test text {first_test_item['text']}")

    # Load the dataloaders
    train_dataloader = myMedicalDataModule.train_dataloader()
    val_dataloader = myMedicalDataModule.val_dataloader()
    test_dataloader = myMedicalDataModule.test_dataloader()

    # Test the dataloaders
    for batch in train_dataloader:
        print(batch['image'].shape)
        print(batch['text'])
        break

    for batch in val_dataloader:
        print(batch['image'].shape)
        print(batch['text'])
        break

    for batch in test_dataloader:
        print(batch['image'].shape)
        print(batch['text'])
        break


if __name__ == "__main__":
    from open_clip import get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
    import matplotlib.pyplot as plt

    _test_ComplexMedicalDataset()
    _test_MyDatamodule()
    print("All tests passed!")