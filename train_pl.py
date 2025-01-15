import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision import transforms
import torchvision.transforms as T

from open_clip import create_model_from_pretrained # works on open-clip-torch>=2.23.0, timm>=0.9.8

from model_pl import CLIPLinearProbe
import config_pl
from dataset_pl import MyDatamodule


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('medium') 

model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

class_descriptions = [
    "Characterized by scattered areas of pattern density",
    "Fatty predominance",
    "Extremely dense",
    "Heterogeneously dense",
    "Moderately dense"
    ]

linear_probe = CLIPLinearProbe(
    model=model, 
    class_descriptions=class_descriptions, 
    learning_rate=config_pl.LEARNING_RATE, 
    weight_decay=config_pl.WEIGHT_DECAY, 
    dropout_rate=config_pl.DROPOUT_RATE,
    )

early_stop_callback = EarlyStopping(
        monitor='val_epoch_loss',  # quantity to monitor
        min_delta=0.00,            # minimum change to qualify as an improvement
        patience=50,               # number of epochs with no improvement after which training will be stopped
        verbose=True,              # enable verbose mode
        mode='min'                 # "min" means lower val_loss is better
    )

logger = TensorBoardLogger(
    save_dir='david_test',
    name='linear_probe',
    default_hp_metric=False
)

trainer = L.Trainer(
    logger=logger,
    accelerator=config_pl.ACCELERATOR,
    devices=config_pl.DEVICES,
    max_epochs=config_pl.NUM_EPOCHS,
    callbacks=[early_stop_callback],
    precision=config_pl.PRECISION
) 

train_transform = T.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(10),
        T.RandomAffine(degrees=(1,10), translate=(0.1, 0.1), scale=(0.9, 1.1))
    ])

test_transform = T.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])

myMedicalDataModule = MyDatamodule(
        data_dir = config_pl.DATA_DIR,
        transforms={'train': train_transform, 'test': test_transform},
        batch_size=config_pl.BATCH_SIZE,
        num_workers=config_pl.NUM_WORKERS)

trainer.fit(model=linear_probe, datamodule=myMedicalDataModule)
