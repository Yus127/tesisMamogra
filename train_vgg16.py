"""
Training script for VGG16 fine-tuning on breast density classification.

Full fine-tuning with differential learning rates for feature extractor
and classifier head.

Usage: python train_vgg16.py
"""
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import torchvision.transforms as T

from models.vgg16_model import VGG16Classifier
import config
from data.dataset import MyDatamodule


def run_main():
    torch.set_float32_matmul_precision('medium')

    class_descriptions = [
        "Fatty predominance",
        "Characterized by scattered areas of pattern density",
        "Heterogeneously dense",
        "Extremely dense"
    ]

    model = VGG16Classifier(
        class_descriptions=class_descriptions,
        learning_rate=config.LEARNING_RATE,
        feature_learning_rate=0.00001,
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
        name='vgg16',
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
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(10),
        T.RandomAffine(degrees=(1, 10), translate=(0.1, 0.1), scale=(0.9, 1.1))
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

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == '__main__':
    run_main()
