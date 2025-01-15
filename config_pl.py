CLIP_HIDDEN_SIZE = 512

# Training hyperparameters
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.01
BATCH_SIZE = 16
NUM_EPOCHS = 2
DROPOUT_RATE = 0.2
PRECISION = "bf16-mixed"

# These are according to the tokenizer
BOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3
PAD_TOKEN_ID = 0
HIDDEN_SIZE = 224

# Dataset
DATASET = "400images"
DATA_DIR = ".data/" + DATASET + "/"
NUM_WORKERS = 19

# Compute related
ACCELERATOR = "gpu"
DEVICES = [0]