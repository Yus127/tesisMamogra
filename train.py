import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision import transforms
import torchvision.transforms as T

from open_clip import create_model_from_pretrained # works on open-clip-torch>=2.23.0, timm>=0.9.8

from model_pl import CLIPLinearProbe
import config
from dataset_pl import MyDatamodule


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('medium') 

model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

class_descriptions = [
    "Characterized by scattered areas of pattern density",
    "Fatty predominance",
    "Extremely dense",
    "Heterogeneously dense"
    ]

linear_probe = CLIPLinearProbe(
    model=model, 
    class_descriptions=class_descriptions, 
    learning_rate=config.LEARNING_RATE, 
    weight_decay=config.WEIGHT_DECAY, 
    dropout_rate=config.DROPOUT_RATE,
    l2_lambda=config.L2_LAMBDA
    )

early_stop_callback = EarlyStopping(
        monitor='val_loss',  # quantity to monitor
        min_delta=0.00,            # minimum change to qualify as an improvement
        patience=10,               # number of epochs with no improvement after which training will be stopped
        verbose=True,              # enable verbose mode
        mode='min'                 # "min" means lower val_loss is better
    )

logger = TensorBoardLogger(
    save_dir='logging_tests',
    name='linear_probe',
    version = "4_balanced_no_augmentation_final_v2",
    default_hp_metric=False
)

trainer = L.Trainer(
    logger=logger,
    accelerator=config.ACCELERATOR,
    devices=config.DEVICES,
    max_epochs=config.NUM_EPOCHS,
    callbacks=[early_stop_callback],
    precision=config.PRECISION,
    log_every_n_steps=25
) 

train_transform = T.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))#,
        #T.RandomVerticalFlip(p=0.5),
        #T.RandomRotation(10),
        #T.RandomAffine(degrees=(1,10), translate=(0.1, 0.1), scale=(0.9, 1.1))
    ])

test_transform = T.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])

myMedicalDataModule = MyDatamodule(
        data_dir = config.DATA_DIR,
        transforms={'train': train_transform, 'test': test_transform},
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS)

trainer.fit(model=linear_probe, datamodule=myMedicalDataModule)
trainer.test(model=linear_probe, datamodule=myMedicalDataModule)
