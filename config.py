# Network architecture configs
INPUT_CHANNELS = 9
OUTPUT_CHANNELS = 71
KERNEL = 3
NUM_RESIDUALS = 19

# Search parameters
C_PUCT = 5
SEARCHES_PER_MOVE = 32

# Training parameters
LEARNING_RATE = .001
EPOCHS = 100_000 #
BATCH_SIZE = 64

# RL params
SELF_PLAY_KEEP_PROB = 0.3
NEW_GAMES_PER_DS = 512
RL_DS_SIZE = NEW_GAMES_PER_DS * 8
NUM_MOVES_MAX_TEMPERATURE = 32
MAX_MOVES = 9*9*2

# Device
import torch
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE  = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
