# Network architecture configs
INPUT_CHANNELS = 9
OUTPUT_CHANNELS = 71
KERNEL = 3
NUM_RESIDUALS = 19

# Search parameters
C_PUCT = 5

# Training parameters
LEARNING_RATE = .001
EPOCHS = 100
BATCH_SIZE = 64

# Device
import torch
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE  = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
