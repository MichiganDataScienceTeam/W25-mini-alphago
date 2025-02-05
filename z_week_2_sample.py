"""
SAMPLE CODE
For reference purposes only
"""

import numpy as np
from numpy.typing import NDArray

from game_node import GameNode

def node_to_tensor(node: GameNode) -> NDArray:
    """
    Converts a GameNode of NxN boards to a NDArray[N, N, 9]
    input for NN

    Args:
        node: The GameNode to convert
    """

    N = node.size
    LOOKBACK = 3 # Generates the 4*2=8 channels, could be any reasonably #

    # First LOOKBACK * 2 channels
    out = []
    curr = node

    for _ in range(LOOKBACK):
        grid = curr.grid

        out.append((grid == 1).astype(np.uint8)) # Black stone grid
        out.append((grid == 2).astype(np.uint8)) # White stone grid

        curr = curr.prev # Move back one

        # Fill the rest with 0s if we can't go back further
        if curr is None:
            out += [np.zeros((N, N)) for __ in range(LOOKBACK*2 - len(out))]
            break
    
    # Last channel (turn matrix)
    if node.move % 2 == 0:
        out.append(np.ones((N, N)))
    else:
        out.append(0 - np.ones((N, N)))
    
    # Stack and return
    return np.stack(out)


if __name__ == "__main__":
    # Testing code to make sure this works
    node = GameNode(9) # init to root node

    # Play very badly
    node = node.create_child((0, 0))
    node = node.create_child((0, 1))
    node = node.create_child((0, 2))
    node = node.create_child((0, 3))
    node = node.create_child((0, 4))

    # Compute numpy tensor
    result = node_to_tensor(node)
    print(result)
    print(result.shape)

