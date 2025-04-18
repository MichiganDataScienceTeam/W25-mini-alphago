from numpy import random

from network import NeuralNet
from monte_carlo import MonteCarlo
from tree_node import TreeNode
from game_node import GameNode

from config import *


class Bot:
    def __init__(self):
        """
        Add whatever member variables you want
        """
        raise(NotImplementedError)
        

    def choose_move(self) -> tuple[int, int]:
        """
        This function takes in the board at the current state, and outputs the move it wants to take as a tuple of ints

        It does not make the move itself.
        """
        raise(NotImplementedError)


class MonteCarloBot(Bot):
    def __init__(self, model: NeuralNet = NeuralNet(), device: str = DEVICE):
        """
        Add whatever member variables you want
        """
        self.model = model
        self.mcts = MonteCarlo(self.model, TreeNode(GameNode(9)), device)
    

    def reset_tree(self) -> None:
        self.mcts.reset()
        self.mcts = MonteCarlo(self.model, TreeNode(GameNode(9)), self.mcts.device)


    def choose_move(self, num_searches: int = 10) -> TreeNode:
        """
        This function takes in the board at the current state, and outputs the move it wants to take as a tuple of ints

        It does not make the move itself.
        """
        
        for _ in range(num_searches):
            self.mcts.search()
            
        probs = self.mcts.curr.get_policy()
        
        if len(probs) == 0:
            return (-1, -1)

        action_index = random.choice(len(probs), p=probs)
        return self.mcts.curr.nexts[action_index]
    

    def make_move(self, move: tuple[int, int]) -> None:
        self.mcts.move_curr(move)