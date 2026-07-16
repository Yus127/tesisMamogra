"""
Configuration file for breast density classification experiments.

This module contains all hyperparameters and settings used across different
experiments (ConvNeXt fine-tuning, BioMedCLIP linear probing, VGG16 baseline).
"""
# Training hyperparameters
LEARNING_RATE = 0.0001 # AdamW learning rate for all models
WEIGHT_DECAY = 0.01 # L2 regularization weight decay (AdamW)
BATCH_SIZE = 64
NUM_EPOCHS = 200  # Note: Early stopping typically triggers around epoch 40
DROPOUT_RATE = 0.01 # Dropout probability in classifier head
L2_LAMBDA = 0.001


DATA_DIR ="/mnt/Pandora/Datasets/MamografiasMex/4kimages/"

# Alternative configuration for relative paths:
# DATASET = "your_dataset_name"
# DATA_DIR = ".data/" + DATASET + "/"

NUM_WORKERS = 19

# Compute related
ACCELERATOR = "gpu"
DEVICES = [0] # GPU device IDs to use (e.g., [0] for single GPU,
# [0, 1] for multi-GPU)
PRECISION = "bf16-mixed" # Options:
# - "32": Full 32-bit precision (slower, more memory)
# - "16-mixed": 16-bit mixed precision (faster, less memory)
# - "bf16-mixed": BF16 mixed precision (recommended for A100/H100)
# - "16": Full 16-bit (not recommended, numerically unstable)



