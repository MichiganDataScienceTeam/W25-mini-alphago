from game_node import GameNode

import torch

from config import *

# TODO: Add function for creating a data tuple from a GameNode

def node_to_tensor(node: GameNode) -> torch.Tensor:
    """
    Converts a GameNode of NxN boards to a Tensor[N, N, 9]
    input

    Args:
        node: The GameNode to convert
    """

    N = node.size
    LOOKBACK = (INPUT_CHANNELS - 1)//2 # Generates INPUT_CHANNELS - 1 channels

    # First LOOKBACK * 2 channels
    out = []
    curr = node

    for _ in range(LOOKBACK):
        grid = torch.tensor(curr.grid)

        out.append((grid == 1).to(torch.float32))
        out.append((grid == 2).to(torch.float32))

        curr = curr.prev

        if curr is None:
            out += [torch.zeros(N, N) for __ in range(LOOKBACK*2 - len(out))]
            break
    
    # Last channel (turn flag)
    if node.move % 2 == 0:
        out.append(torch.ones(N, N))
    else:
        out.append(-1 * torch.ones(N, N))
    
    # Stack and return
    return torch.stack(out)

