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

tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
# Initialize BioMedCLIP model and preprocessor
model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
print(dir(model))

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



"""
lightning_model = LightningBiomedCLIP(
    model=model,
    tokenizer=tokenizer,
    clip_hidden_size = config_pl.CLIP_HIDDEN_SIZE,
    learning_rate =config_pl.LEARNING_RATE,
    weight_decay=config_pl.WEIGHT_DECAY,
    warmup_steps = config_pl.WARMUP_STEPS,
    hidden_size=config_pl.HIDDEN_SIZE,
    vocab_size=tokenizer.tokenizer.vocab_size,
    max_length=config_pl.MAX_LENGHT,
    bos_token_id=config_pl.BOS_TOKEN_ID,  
    eos_token_id=config_pl.EOS_TOKEN_ID,
    pad_token_id=config_pl.PAD_TOKEN_ID
    
)
"""
#print(lightning_model.model)


# Create dataset instance
dataset = ComplexMedicalDataset(
    data_dir="/Users/YusMolina/Documents/tesis/biomedCLIP/data/datosMex/images/train.json",
    #data_dir="/home/yus/test/tesisMamogra/train.json",
    processor=model,
    tokenizer=tokenizer
)


dataval = ComplexMedicalDataset(
    data_dir="/Users/YusMolina/Documents/tesis/biomedCLIP/data/datosMex/images/test.json",
    #data_dir="/home/yus/test/tesisMamogra/test.json",
    processor=model,
    tokenizer=tokenizer
)


print(f"Sample from dataset: {dataset[0]}")

if torch.any(dataset[4]["image"] != 0):
    print("Tensor contains non-zero values.")
else:
    print("Tensor is full of zeros.")

# Create DataLoader
train_loader = DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True, 
    collate_fn=ComplexMedicalDataset.collate_fn
    )

val_loader = DataLoader(
    dataval, 
    batch_size=32, 
    shuffle=True, 
    collate_fn=ComplexMedicalDataset.collate_fn
    )

print(f"DataLoader configuration: {train_loader}")

linear_probe = CLIPLinearProbe(model, class_descriptions, tokenizer, preprocess, True)


from pytorch_lightning.loggers import TensorBoardLogger

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


#trainer.tune, find the hpyerparameters

trainer.fit(linear_probe, train_dataloaders=train_loader, val_dataloaders=val_loader)

