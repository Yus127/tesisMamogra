# imports
import torch 
import torch.nn as nn
import torch.optim as optim #optimization algorithms 
import torch.nn.functional as F # functions with no parameters relu, tanh
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms # transformations in the dataset
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import random_split
import torchmetrics
from torchmetrics import Metric

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)
        assert preds.shape == target.shape
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()

class NN(pl.LightningModule):
    def __init__(self,  input_size, num_classes): #28 x 28 
        super().__init__() #cause the initialization of the parent 
        self.fc1 = nn.Linear(input_size, 50) #hideen layer of 50 nodes
        self.fc2 = nn.Linear(50, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy= torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_Score= torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.my_accuracy = MyAccuracy()

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy =self.my_accuracy(scores, y)
        f1_score = self.f1_Score(scores, y)
        self.log_dict({"train_loss": loss ,"train_accuracy": accuracy, "train_f1_score": f1_score}, on_step=False, on_epoch=True, prog_bar=True)
        
        #self.log("train_loss", loss)
        return {"loss": loss, "scores":scores, "y":y}
    
    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    #def training_epoch_end(self, outputs):#compute average loss, etc

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch 
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)
        # we have predict step, outputs, argmax and thats the prediction
    def predict_step(self, batch, batch_idx):
        x, y = batch 
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds


def main():
    input_size = 784
    num_classes = 10
    batch_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NN(input_size=input_size, num_classes=num_classes).to(device)
    dm = MnistDataModule(data_dir="dataset/", batch_size=batch_size, num_workers=4)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        min_epochs=1,
        max_epochs=3,
        precision=16
    ) #num_nodes
    #trainer.tune, find the hpyerparameters
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)

if __name__ == '__main__':
    main()
