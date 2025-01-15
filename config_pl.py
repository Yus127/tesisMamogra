# Training hyperparameters
#INPUT_SIZE = 784
#NUM_CLASSES = 10
CLIP_HIDDEN_SIZE = 512
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.01
BATCH_SIZE = 8
NUM_EPOCHS = 500
DROPOUT_RATE = 0.2
MAX_LENGHT = 64
# These are according to the tokenizer
BOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3
PAD_TOKEN_ID = 0
HIDDEN_SIZE = 224

# Dataset
DATA_DIR = ".data/"
NUM_WORKERS = 19

# Compute related
ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = '16-mixed'