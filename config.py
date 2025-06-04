# Training hyperparameters
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.01
BATCH_SIZE = 64
NUM_EPOCHS = 200
DROPOUT_RATE = 0.01
L2_LAMBDA = 0.001

#DATA_DIR = ".data/" + DATASET + "/"
DATA_DIR ="/mnt/Pandora/Datasets/MamografiasMex/4kimages/"

NUM_WORKERS = 19

# Compute related
ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = "bf16-mixed"
