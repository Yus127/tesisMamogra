"""BioMedCLIP linear probing — frozen encoder + trainable linear head.

Usage: python train_clip.py
"""
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import torchvision.transforms as T

from open_clip import create_model_from_pretrained

from models.clip_probe import CLIPLinearProbe
import config
from data.dataset import MyDatamodule


def run_main():
    torch.set_float32_matmul_precision('medium')

    model, preprocess = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )

    # Paper uses 4 standard BI-RADS classes; 5th class is experimental
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
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        dropout_rate=config.DROPOUT_RATE,
        l2_lambda=config.L2_LAMBDA
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode='min'
    )

    logger = TensorBoardLogger(
        save_dir='logging_tests',
        name='linear_probe',
        version='4_unbalanced_no_augmentation',
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
        T.ToTensor(),
        T.Resize((224, 224)),
        # T.RandomVerticalFlip(p=0.5),
        # T.RandomRotation(10),
        # T.RandomAffine(degrees=(1, 10), translate=(0.1, 0.1), scale=(0.9, 1.1))
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224))
    ])

    datamodule = MyDatamodule(
        data_dir=config.DATA_DIR,
        transforms={'train': train_transform, 'test': test_transform},
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )

    trainer.fit(model=linear_probe, datamodule=datamodule)
    trainer.test(model=linear_probe, datamodule=datamodule)


if __name__ == '__main__':
    run_main()
