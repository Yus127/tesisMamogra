# Training hyperparameters
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
BATCH_SIZE = 32
NUM_EPOCHS = 60
DROPOUT_RATE = 0.1
L2_LAMBDA = 0.00

# Tokenizer settings
BOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3
PAD_TOKEN_ID = 0
HIDDEN_SIZE = 224

DATASET = "400images"
#DATA_DIR = ".data/" + DATASET + "/"
DATA_DIR ="/mnt/Pandora/Datasets/MamografiasMex/4kimages/"

NUM_WORKERS = 19

# Compute related
ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = "bf16-mixed"
