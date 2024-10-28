from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms # transformations in the dataset
import pytorch_lightning as pl
from torch.utils.data import random_split

#dataModule
class MnistDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def prepare_data(self): #download the data, tokenize the text 
        #here it is the customDataset()
        #single gpu
        #my_ds = CustomDataset(train_csv)
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage):
        #multiple gpu
        entire_dataset = datasets.MNIST(root=self.data_dir, train=True, transform = transforms.ToTensor(), download=False)
        self.train_ds,self.val_ds =random_split(entire_dataset,[50000,10000])
        self.test_ds =datasets.MNIST(root= self.data_dir, train=False, transform=transforms.ToTensor(), download=False)
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers = self.num_workers, shuffle=False)
        
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers = self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers = self.num_workers, shuffle=False)

