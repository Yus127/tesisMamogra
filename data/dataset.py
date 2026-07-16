"""
Custom dataset and PyTorch Lightning DataModule for loading mammographic images
paired with radiology reports. Handles:
    - Loading images and text from JSON metadata files
    - Applying transformations and augmentations
    - Splitting data into train/validation/test sets
    - Creating efficient DataLoaders with parallel workers

Dataset Structure:
    The expected JSON format contains image paths and corresponding radiology reports:
    [
        {
            "filename": "S0018466_2016_LMLO.tif",
            "report": "Heterogeneously dense",
            "image_path": "4kimages/S0018466_2016_LMLO.tif"
        },
        ...
    ]
"""
import json
import os

import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import config
import lightning as L
import matplotlib.pyplot as plt


class ComplexMedicalDataset(Dataset):
    """
    Loads mammography images along with their corresponding textual density reports
    from JSON metadata files (train.json / test.json).
    """

    def __init__(self, data_dir: str, train: bool = True, transform=None):
        super(ComplexMedicalDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform

        json_file = "train.json" if train else "test.json"
        json_path = os.path.join(self.data_dir, json_file)

        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"{json_file} not found in {self.data_dir}. "
                f"Expected path: {json_path}"
            )

        with open(json_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        image_path = item["filename"]
        image = cv2.imread(os.path.join(self.data_dir, image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found at {os.path.join(self.data_dir, image_path)}")

        # Convert BGR (OpenCV default) to RGB expected by all pretrained models
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        text = item['report']

        return {"image": image, "text": text}


class MyDatamodule(L.LightningDataModule):
    """
    Handles train.json and test.json loading, 85/15 train/val split,
    and DataLoader creation.
    """

    def __init__(self, data_dir: str, transforms: dict, batch_size: int = 32, num_workers: int = 1):
        super(MyDatamodule, self).__init__()
        self.data_dir = data_dir
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        try:
            training_data = ComplexMedicalDataset(
                data_dir=self.data_dir,
                train=True,
                transform=self.transforms['train']
            )
            self.train_dataset, self.validation_dataset = torch.utils.data.random_split(
                training_data,
                [0.85, 0.15],
                generator=torch.Generator().manual_seed(42)
            )
        except FileNotFoundError as e:
            print(e)
            self.train_dataset = None
            self.validation_dataset = None

        try:
            self.test_dataset = ComplexMedicalDataset(
                data_dir=self.data_dir,
                train=False,
                transform=self.transforms['test']
            )
        except FileNotFoundError as e:
            print(e)
            self.test_dataset = None

    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("No training dataset found.")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.validation_dataset is None:
            raise ValueError("No validation dataset found.")
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("No test dataset found.")
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


# ── quick smoke tests ─────────────────────────────────────────────────────────

def _test_ComplexMedicalDataset():
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])
    dataset = ComplexMedicalDataset(data_dir=config.DATA_DIR, transform=train_transform)
    first_item = dataset.__getitem__(0)
    print(f"First item shape: {first_item['image'].shape}")
    plt.imshow(first_item['image'].squeeze().permute(1, 2, 0))
    plt.show()
    print(first_item['text'])


def _test_MyDatamodule():
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])
    dm = MyDatamodule(
        data_dir=config.DATA_DIR,
        transforms={'train': train_transform, 'test': train_transform},
        batch_size=2,
        num_workers=1
    )
    dm.setup()

    for split, ds in [('train', dm.train_dataset), ('val', dm.validation_dataset), ('test', dm.test_dataset)]:
        if ds is None:
            print(f"No {split} dataset found.")
        else:
            item = ds.__getitem__(0)
            print(f"{split} item shape: {item['image'].shape}, text: {item['text']}")


if __name__ == "__main__":
    _test_MyDatamodule()
    print("All tests passed!")
