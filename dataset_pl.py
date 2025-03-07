import json
import os

import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import config
import lightning as L

"""Expected JSON Structure
[
    {
        "image_path": "<dataset>/<filename.tif>",
        "filename": "<filename.tif>",
        "report": "Medical report text."
    }
]
"""
class ComplexMedicalDataset(Dataset):
    def __init__(self, data_dir:str, train:bool=True, transform=None):
        super(ComplexMedicalDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform

        if train:
            # Check there is a train.json file
            if not os.path.exists(os.path.join(self.data_dir, "train_balanced.json")):
                raise FileNotFoundError("train.json file not found in the data directory.")
            with open(os.path.join(data_dir, "train_balanced.json"), 'r') as f:
                self.data = json.load(f)
        else:
            # Check there is a test.json file
            if not os.path.exists(os.path.join(self.data_dir, "test_balanced.json")):
                raise FileNotFoundError("test.json file not found in the data directory.")
            with open(os.path.join(data_dir, "test_balanced.json"), 'r') as f:
                self.data = json.load(f)

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        
        # Load image
        image_path = item["filename"]
        image = cv2.imread(os.path.join(self.data_dir, image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found at {os.path.join(self.data_dir, image_path)}")

        if self.transform:
            image = self.transform(image)

        # Load text
        text = item['report']
        
        return {"image": image, "text": text}


class MyDatamodule(L.LightningDataModule):
    def __init__(self, data_dir:str, transforms:dict, batch_size:int, num_workers:int):
        super(MyDatamodule, self).__init__()

        # Dataset info
        self.data_dir = data_dir
        self.transforms = transforms
        # Dataloaders info
        self.batch_size = batch_size
        self.num_workers = num_workers


    def setup(self, stage=None):
        '''
        Executes on every GPU. Setup the dataset for training, validation and testing.
        '''
        # Load all training data
        try:
            training_data = ComplexMedicalDataset(
                data_dir=self.data_dir,
                train=True,
                transform=self.transforms['train']
            )
            # Split training data into training and validation
            self.train_dataset, self.validation_dataset = torch.utils.data.random_split(training_data, [0.85, 0.15])
        except FileNotFoundError as e:
            print(e)
            print("Skipping training data.")
            self.train_dataset = None
            self.validation_dataset = None

        # Load test data
        try:
            self.test_dataset = ComplexMedicalDataset(
                data_dir=self.data_dir,
                train=False,
                transform=self.transforms['test']
            )
        except FileNotFoundError as e:
            print(e)
            print("Skipping test data.")
            self.test_dataset = None

    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("No training dataset found.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        if self.validation_dataset is None:
            raise ValueError("No validation dataset found.")
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("No test dataset found.")
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

    myMedicalDataset = ComplexMedicalDataset(data_dir=config.DATA_DIR+"400images/", transform=train_transform)

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

    myMedicalDataModule = MyDatamodule(
        data_dir=config.DATA_DIR,
        transforms={'train': train_transform, 'test': train_transform},
        batch_size=2,
        num_workers=1)

    myMedicalDataModule.setup()

    if myMedicalDataModule.train_dataset is None:
        print("No training dataset found.")
    else:
        first_train_item = myMedicalDataModule.train_dataset.__getitem__(0)
         # Visualize the first image of each dataset
        print(f"First train item shape: {first_train_item['image'].shape}")
        plt.imshow(first_train_item["image"].squeeze().permute(1, 2, 0))
        plt.show()
        print(f"First train text {first_train_item['text']}")
    
    if myMedicalDataModule.validation_dataset is None:
        print("No validation dataset found.")
    else:
        first_val_item = myMedicalDataModule.validation_dataset.__getitem__(0)
        print(f"First val item shape: {first_val_item['image'].shape}")
        plt.imshow(first_val_item["image"].squeeze().permute(1, 2, 0))
        plt.show()
        print(f"First val text {first_val_item['text']}")

    if myMedicalDataModule.test_dataset is None:
        print("No test dataset found.")
    else:
        first_test_item = myMedicalDataModule.test_dataset.__getitem__(0)
        print(f"First test item shape: {first_test_item['image'].shape}")
        plt.imshow(first_test_item["image"].squeeze().permute(1, 2, 0))
        plt.show()
        print(f"First test text {first_test_item['text']}")

    # Load the dataloaders
    try:
        # Initialize the dataloaders
        train_dataloader = myMedicalDataModule.train_dataloader()
        for batch in train_dataloader:
            print(f"Batch shape retrieved by train dataloader : {batch['image'].shape}")
            print(f"Batch text retrieved by train dataloader: {batch['text']}")
            break
    except ValueError as e:
        print(e)

    try:
        val_dataloader = myMedicalDataModule.val_dataloader()
        for batch in val_dataloader:
            print(f"Batch shape retrieved by val dataloader : {batch['image'].shape}")
            print(f"Batch text retrieved by val dataloader: {batch['text']}")
            break
    except ValueError as e:
        print(e)

    try:
        test_dataloader = myMedicalDataModule.test_dataloader()
        for batch in test_dataloader:
            print(f"Image shape retrieved by test dataloader : {batch['image'].shape}")
            print(f"Batch text retrieved by test dataloader: {batch['text']}")
            break
    except ValueError as e:
        print(e)
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    #_test_ComplexMedicalDataset()
    #print("ComplexMedical Dataset test passed!")
    _test_MyDatamodule()
    print("MyDatamodule test passed!")

    print("All tests passed!")
