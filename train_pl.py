import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from PIL import Image


import nrrd
from torchvision import transforms
from typing import List, Optional

from model_pl import LightningBiomedCLIP
from dataset_pl import ComplexMedicalDataset
import config_pl

from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
# Initialize BioMedCLIP model and preprocessor
model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
#print(dir(tokenizer))


lightning_model = LightningBiomedCLIP(
    model=model,
    tokenizer=tokenizer,
    vocab_size=28895,
    max_length=64,
    bos_token_id=2,  # Adjust these token IDs based on your tokenizer
    eos_token_id=3,
    pad_token_id=0,
    hidden_size=224
)
#print(lightning_model.model)

# Initialize tokenizer
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

# Create dataset instance
dataset = ComplexMedicalDataset(
    data_dir="/Users/YusMolina/Documents/tesis/biomedCLIP/data/datosMex/",
    processor=model,
    tokenizer=tokenizer
)
print(dataset)
print("dataset")

#print(f"Sample from dataset: {dataset[4]}")

if torch.any(dataset[4]["image"] != 0):
    print("Tensor contains non-zero values.")
else:
    print("Tensor is full of zeros.")

# Create DataLoader
dataloader = DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True, 
    collate_fn=ComplexMedicalDataset.collate_fn
    )

print(f"DataLoader configuration: {dataloader}")

trainer = pl.Trainer(
    accelerator=config_pl.ACCELERATOR,
    devices=config_pl.DEVICES,
    min_epochs=1,
    max_epochs=config_pl.NUM_EPOCHS,
    precision=config_pl.PRECISION,
    log_every_n_steps = 1
) 
#trainer.tune, find the hpyerparameters

trainer.fit(lightning_model, dataloader)
#trainer.validate(model, dm)
#trainer.test(model, dm)
