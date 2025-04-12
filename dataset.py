import os
import torch

from game_node import GameNode
from imported_game import ImportedGame
from data_preprocess import node_to_tensor, one_hot_policy

from typing import Tuple


class Dataset:
    """
    Dataset to interface game data with PyTorch DataLoader
    """

    def __init__(self):
        self.positional_data = []
        self.start_indices = []


    def __len__(self):
        return len(self.positional_data)
    

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.positional_data[idx]


    def load_dir(self, game_directory: str) -> None:
        """
        Loads all files in specified directory assuming all files
        are valid Go games in SGF format

        Args:
            game_directory: path to the directory to load from
        """

        dirs = os.listdir(game_directory)

        for i, game_file in enumerate(dirs):
            print(f"[{i+1}/{len(dirs)}] Loading game {game_file}")
            filepath = os.path.join(game_directory, game_file)
            this_game = ImportedGame(filepath)

            # Iterate through the game
            node = this_game.linked_list()
            final_eval = this_game.meta.get("final_eval")

            self.add_game(node, final_eval)


    def add_game(self, node: GameNode, final_eval: float) -> None:
        """
        Adds a game to the dataset

        Args:
            node: the last GameNode in the game
            final_eval: the final eval {-1, 1} of the game
        """

        self.start_indices.append(len(self.positional_data))

        last_human_policy = one_hot_policy(node)
        node = node.prev

        while node is not None:
            s = node_to_tensor(node)
            z = torch.tensor([final_eval], dtype=torch.float32)
            pi = last_human_policy
            last_human_policy = one_hot_policy(node)
            self.positional_data.append((s, z, pi))

            node = node.prev


    def remove_first_n(self, n) -> None:
        """
        Removes the first (order of insertion) n games in the dataset

        Args:
            n: the number of games to remove
        """

        self.start_indices = self.start_indices[n:]
        start_index = self.start_indices[0]
        self.start_indices = [i - start_index for i in self.start_indices]
        self.positional_data = self.positional_data[start_index:]

