import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import nrrd
from typing import List, Optional
from lightning.pytorch import Trainer

from model_pl import LightningBiomedCLIP, CLIPLinearProbe
from dataset_pl import ComplexMedicalDataset
import config_pl
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
from pytorch_lightning.loggers import TensorBoardLogger

tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
# Initialize BioMedCLIP model and preprocessor
model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
#print(dir(model))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class_descriptions = ['Characterized by scattered areas of pattern density',
       'Extremely dense', 'Heterogeneously dense', 'Fatty predominance',
       'Moderately dense']

early_stop_callback = EarlyStopping(
        monitor='train_loss',      # quantity to monitor
        min_delta=0.00,          # minimum change to qualify as an improvement
        patience=50,              # number of epochs with no improvement after which training will be stopped
        verbose=True,            # enable verbose mode
        mode='min'               # "min" means lower val_loss is better
    )


linear_probe = CLIPLinearProbe(model, class_descriptions, tokenizer, preprocess, True)



#logger = TensorBoardLogger("lightning_logs", name="clip_probe")
logger = TensorBoardLogger(
    save_dir='lightning_logs',
    name='clip_probe',
    default_hp_metric=False
)
trainer = pl.Trainer(
    logger=logger,
    accelerator=config_pl.ACCELERATOR,
    devices=config_pl.DEVICES,
    min_epochs=1,
    max_epochs=config_pl.NUM_EPOCHS,
    precision=config_pl.PRECISION,
    log_every_n_steps = 3,
    callbacks=[early_stop_callback]
    #deterministic=True

) 

train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])

myMedicalDataModule = MyDatamodule(
        data_dir = os.getenv("DATA_DIR"),
        tokenizer=tokenizer,
        transforms={'train': train_transform, 'test': train_transform},
        batch_size=32,
        num_workers=19)

#trainer.tune, find the hpyerparameters

trainer.fit(linear_probe, myMedicalDataModule)
