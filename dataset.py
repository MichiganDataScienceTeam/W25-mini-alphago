from imported_game import ImportedGame
import numpy as np
import os
from data_preprocess import node_to_tensor


def human_target_policy(gameNode):
    """Return a board array with a 1 where the human player moved"""
    target = np.zeros(gameNode.size * gameNode.size + 1, dtype=float)

    # handle pass
    if gameNode.prev_move == (-1, -1) or gameNode.prev_move is None:
        target[-1] = 1

    # handle move
    else:
        row = gameNode.prev_move[0]
        col = gameNode.prev_move[1]
        target[gameNode.size * row + col] = 1

    return target


class Dataset:
    """
    Dataset to interface game data with PyTorch DataLoader
    """

    def __init__(self, game_directory):
        self.positional_data = []
        self.load_games(game_directory)
        self.ss, self.zs, self.pis = list(zip(*self.positional_data))

    def __len__(self):
        return len(self.positional_data)
    
    def __getitem__(self, idx):
        return self.positional_data[idx]

    def load_games(self, game_directory):
        for i, game_file in enumerate(dirs := os.listdir(game_directory)):
            print(f"[{i}/{len(dirs)}] Loading game {game_file}")
            filepath = os.path.join(game_directory, game_file)
            this_game = ImportedGame(filepath)

            # Iterate through the game
            node = this_game.linked_list()
            last_human_policy = human_target_policy(node)

            while node is not None:
                s = node_to_tensor(node)
                z = this_game.meta.get("final_eval")
                pi = last_human_policy
                last_human_policy = human_target_policy(node)
                self.positional_data.append((s, z, pi))

                node = node.prev


