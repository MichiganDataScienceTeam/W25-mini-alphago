import numpy as np
import random

from typing import Tuple

from board import Board
from network import NeuralNet
from monte_carlo import MonteCarlo
from tree_node import TreeNode
from game_node import GameNode
from data_preprocess import node_to_tensor

from config import *


class Bot:
    def __init__(self):
        """
        Initializes the bot
        """

        self.curr = TreeNode(GameNode(9))
        

    def choose_move(self) -> Tuple[int, int]:
        """
        Chooses a move and returns the location tuple of the move
        """

        raise(NotImplementedError)
    
    
    def reset(self) -> None:
        """
        Resets internal bot data
        """

        self.curr = TreeNode(GameNode(9))


    def register_move(self, loc: Tuple[int, int]) -> None:
        """
        Updates internal bot data with the latest move

        Args:
            loc: the location of the move to play
        """

        self.curr = self.curr.create_child(loc)
    

    def get_tree(self) -> TreeNode:
        return self.curr
    
    def __int__(self) -> int:
        return int(self.curr)


class RandomBot(Bot):
    def choose_move(self) -> Tuple[int, int]:
        """
        Randomly select a move
        """

        next_moves = self.curr.available_moves()
        return random.choice(next_moves)


class SupervisedLearningBot(Bot):
    def __init__(self, model):
        super().__init__()

        self.model = model


    def choose_move(self) -> Tuple[int, int]:
        """
        Sample the move from the available allowed actions using the model's prior
        """
        
        input_tensor = node_to_tensor(self.curr).unsqueeze(0)

        prior, value = self.model.forward(input_tensor)
        next_moves = self.curr.available_moves()

        #Don't need tensor in a batch or for computation, just want to sample a single thing from it
        prior = prior[0].detach().cpu().numpy()

        action_index = np.random.choice(len(prior), p=prior)

        return next_moves[action_index % len(next_moves)]


class MonteCarloBot(Bot):
    def __init__(self, model: NeuralNet = None, device: str = DEVICE, always_allow_pass: bool = False):
        """
        Initializes the bot

        Args:
            model: the NeuralNet model for MCTS
            device: the device to run NN computations on
            always_allow_pass: whether to always allow pass
        """

        if model is None:
            model = NeuralNet()

        self.model = model
        self.mcts = MonteCarlo(self.model, TreeNode(GameNode(9)), device, always_allow_pass)

        self.mcts.search()
    

    def reset(self) -> None:
        """
        Resets internal bot data
        """

        self.mcts.reset()
        self.mcts = MonteCarlo(self.model, TreeNode(GameNode(9)), self.mcts.device, always_allow_pass=True)


    def choose_move_tree(self, num_searches: int = 10) -> TreeNode:
        """
        Chooses a move and returns the resulting TreeNode

        Args:
            num_searches: number of searches before choosing a move
        """

        for _ in range(num_searches):
            self.mcts.search()

        probs = self.mcts.curr.get_policy()

        if len(probs) == 0:
            return None

        action_index = np.random.choice(len(probs), p=probs)

        assert((-1, -1) in [x.prev_move for x in self.mcts.curr.nexts])

        return self.mcts.curr.nexts[action_index]


    def choose_move(self, num_searches: int = 10) -> Tuple[int, int]:
        """
        Chooses a move and returns the location tuple of the move

        Args:
            num_searches: number of searches before choosing a move
        """

        node = self.choose_move_tree(num_searches)

        if node is None:
            return (-1, -1)
        
        return node.prev_move


    def register_move(self, loc: Tuple[int, int]) -> None:
        """
        Updates internal bot data with the latest move

        Args:
            loc: the location of the move to play
        """

        self.mcts.search()
        self.mcts.move_curr(loc)


    def get_tree(self) -> TreeNode:
        return self.mcts.curr

    def __int__(self) -> int:
        return int(self.mcts.curr)

