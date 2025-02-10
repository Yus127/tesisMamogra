# Training hyperparameters
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.01
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.1
L2_LAMBDA = 0.001

#DATA_DIR = ".data/" + DATASET + "/"
DATA_DIR ="/mnt/Pandora/Datasets/MamografiasMex/4kimages/"

NUM_WORKERS = 19

# Compute related
ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = "bf16-mixed"
