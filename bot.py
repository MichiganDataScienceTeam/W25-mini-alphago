from board import Board
from network import NeuralNet
from monte_carlo import MonteCarlo
from tree_node import TreeNode
from game_node import GameNode
from numpy import random
import json

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
    def __init__(self, model: NeuralNet = NeuralNet()):
        """
        Add whatever member variables you want
        """
        self.model = model
        self.mcts = MonteCarlo(self.model, TreeNode(GameNode(9)))
            

    def choose_move(self, num_searches: int = 10) -> tuple[int, int]:
        """
        This function takes in the board at the current state, and outputs the move it wants to take as a tuple of ints

        It does not make the move itself.
        """
        
        for i in range(num_searches):
            self.mcts.search()
            
        probs = self.mcts.curr.get_policy(temperature=1.0)
        
        if len(probs) == 0:
            return (-1, -1)
            
        action_index = random.choice(len(probs), p=probs)
        move = self.mcts.curr.nexts[action_index].prev_move
        
        return move
        # poss_moves = board.available_moves()
    
    def make_move(self, move: tuple[int, int]) -> None:
        self.mcts.move_curr(move)