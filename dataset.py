import os
import torch

import random

from game_node import GameNode
from tree_node import TreeNode
from imported_game import ImportedGame
from data_preprocess import node_to_tensor, one_hot_policy
from config import *

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

            self.add_sl_game(node, final_eval)


    def add_sl_game(self, node: GameNode, final_eval: float) -> None:
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


    def add_rl_game(self, node: TreeNode, final_eval: float, keep_prob:float = SELF_PLAY_KEEP_PROB) -> None:
        """
        Adds a game to the dataset

        Args:
            node: the last TreeNode in the game
            final_eval: the final eval {-1, 1} of the game
        """

        self.start_indices.append(len(self.positional_data))

        while node is not None:
            s = node_to_tensor(node)
            z = torch.tensor([final_eval], dtype=torch.float32)
            probs = node.get_policy()

            BOARD_SIZE = node.size
            pi = torch.zeros(BOARD_SIZE * BOARD_SIZE + 1)

            for i, child in enumerate(node.nexts):
                prev_move = child.prev_move
                if prev_move == (-1, -1):
                    pi[BOARD_SIZE * BOARD_SIZE] = probs[i].item()
                else:
                    pi[prev_move[0] * BOARD_SIZE + prev_move[1]] = probs[i].item()

            if random.random() < keep_prob:
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

    def save(self, filepath: str) -> None:
        """
        Saves the dataset to a file using PyTorch's serialization

        Args:
            filepath: destination file path to save the dataset
        """
        torch.save({
            'positional_data': self.positional_data,
            'start_indices': self.start_indices
        }, filepath)
        print(f"Dataset saved to {filepath}")

    def load(self, filepath: str) -> None:
        """
        Loads the dataset from a file

        Args:
            filepath: source file path to load the dataset from
        """
        data = torch.load(filepath)
        self.positional_data = data['positional_data']
        self.start_indices = data['start_indices']
        print(f"Dataset loaded from {filepath}")

