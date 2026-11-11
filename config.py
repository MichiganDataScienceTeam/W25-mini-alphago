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
EPOCHS = 100_000
BATCH_SIZE = 64

# SL params
SL_EPOCHS = 20
SL_VAL_FRACTION = 0.1  # Fraction of games/ held out for validation (in addition to val_games/)
SL_AUGMENT = True  # Train on all 8 symmetries of each game

# RL params
SELF_PLAY_KEEP_PROB = 0.2
NEW_GAMES_PER_DS = 256  # Not fully implemented for mp: Must be divisible by PROCESSES
RL_DS_SIZE = NEW_GAMES_PER_DS * 2
NUM_MOVES_ALLOW_PASS = 40  # Most 9x9 pro games last 36 to 53 moves, so passing earlier is almost always a mistake
NUM_MOVES_MAX_TEMPERATURE = 32
MAX_MOVES = 9*9*2
DIRICHLET_ALPHA = 0.15  # Root noise for self-play exploration (about 10 / number of moves, like KataGo)
DIRICHLET_FRACTION = 0.25  # Weight of the noise in the root priors (same as AlphaGo Zero)
RL_INIT_WEIGHTS = "SL_weights.pt"  # Weights to start RL from (None for random, RL_weights.pt to resume)
RL_CHECKPOINT_EVERY = 25  # Epochs between saved copies of the weights (to roll back if training breaks)

# Multiprocessing params
PROCESSES = 16
WORKERS_PER_PROCESS = 3  # Self-play workers per process when using the inference server

# Device
import torch
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE  = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Batch self-play network evaluations on the GPU (see inference_server.py)
# MPS is too slow for small batches, so only use it with CUDA
INFERENCE_SERVER = DEVICE.type == "cuda"
