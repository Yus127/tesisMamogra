# Training hyperparameters
#INPUT_SIZE = 784
#NUM_CLASSES = 10
CLIP_HIDDEN_SIZE = 512
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 1000
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
MAX_LENGHT = 64
# These are according to the tokenizer
BOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3
PAD_TOKEN_ID = 0
HIDDEN_SIZE = 224

# Dataset
#DATA_DIR = "dataset/"
NUM_WORKERS = 8

# Compute related
ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = 16