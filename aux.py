import torch
import lightning as L
from torchvision import transforms
from dataset_pl import MyDatamodule
from open_clip import create_model_from_pretrained, get_tokenizer


# Simple torch cnn
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = torch.nn.Linear(64 * 6 * 6, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 6 * 6)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class LModule(L.LightningModule):
    def __init__(self):
        super(LModule, self).__init__()
        self.model = SimpleCNN()
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    def train_dataloader(self):
        return datamodule.train_dataloader()

tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])

model = LModule()
datamodule = MyDatamodule(data_dir=".data/", tokenizer=tokenizer, transforms={"train":train_transform, "test":train_transform}, batch_size=2, num_workers=1)

trainer = L.Trainer(min_epochs=1,
                    accelerator="cpu",
                    max_epochs=3)

trainer.fit(model=model, datamodule=datamodule)