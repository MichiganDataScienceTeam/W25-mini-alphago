import numpy as np
import random

from network import NeuralNet
from monte_carlo import MonteCarlo
from tree_node import TreeNode
from game_node import GameNode
from data_preprocess import node_to_tensor

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


class RandomBot(Bot):
    def __init__(self):
        pass        

    def choose_move(self, game) -> tuple[int, int]:
        """
        Randomly select a move
        """
        next_moves = game.available_moves()
        return random.choice(next_moves)

    def reset_tree(self, new_game):
        print(f"old tree {self.game}")
        self.game = new_game
        print(f"new tree {self.game}")


class SupervisedLearningBot(Bot):
    def __init__(self, model):
        self.model = model

    def choose_move(self, game) -> tuple[int, int]:
        """
        Sample the move from the available allowed actions using the model's prior
        """
        
        input_tensor = node_to_tensor(game).unsqueeze(0)

        prior, value = self.model.forward(input_tensor)
        next_moves = game.available_moves()

        #Don't need tensor in a batch or for computation, just want to sample a single thing from it
        prior = prior[0].detach().cpu().numpy() 
    
        action_index = np.random.choice(len(prior), p=prior)

    
        return next_moves[action_index % len(next_moves)]

    def reset_tree(self, new_game):
        print(f"old tree {self.game}")
        self.game = new_game
        print(f"new tree {self.game}")


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


    def choose_move(self, num_searches: int = 10, game = None) -> TreeNode:
        """
        This function takes in the board at the current state, and outputs the move it wants to take as a tuple of ints

        It does not make the move itself.
        """
        
        if game == None:
            for _ in range(num_searches):
                self.mcts.search()
                
            probs = self.mcts.curr.get_policy()
            
            if len(probs) == 0:
                return (-1, -1)

            action_index = np.random.choice(len(probs), p=probs)
            return self.mcts.curr.nexts[action_index]
        
        #This is not good code design, but basically if a game is being passed in, we're using the elo calculator, and want to reutrn a move instead
        else:
            self.mcts = MonteCarlo(self.model, game)
            for _ in range(num_searches):
                self.mcts.search()
                
            prior = self.mcts.curr.get_policy()
            next_moves = game.available_moves()

            #Don't need tensor in a batch or for computation, just want to sample a single thing from it
            prior = prior.detach().cpu().numpy() 
        
            action_index = np.random.choice(len(prior), p=prior)

        
            return next_moves[action_index % len(next_moves)]

    def make_move(self, move: tuple[int, int]) -> None:
        self.mcts.move_curr(move)