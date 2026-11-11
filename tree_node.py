from __future__ import annotations
from game_node import GameNode
from typing import List, Self

import numpy as np
import torch

from config import *

class TreeNode(GameNode):
    """
    GameNode wrapper for MCTS

    Args:
        gn: The GameNode to copy from
        num_visits: initial visit count
        total_value: initial total value
        prior: output from the policy head (should be const)
    """


    # Board attributes that lazy children (see get_children) create on first access
    BOARD_ATTRS = {"size", "move", "seen", "groups", "grid", "num_passes", "komi", "index"}


    def __init__(self, gn: GameNode, num_visits = 0, total_value = 0, prior = 0):
        self.__dict__.update(gn.__dict__)

        self.num_visits = num_visits
        self.total_value = total_value
        self.prior = prior


    def __getattr__(self, name):
        """
        Creates the board of a lazy child the first time one of its
        board attributes is accessed (only called if normal lookup fails)
        """

        if name not in TreeNode.BOARD_ATTRS or "prev" not in self.__dict__:
            raise AttributeError(name)

        self.create_board()

        return self.__dict__[name]


    def __str__(self):
        return f"""
        Num visits: {self.num_visits}
        Total value: {self.total_value}
        Prior: {self.prior}
        Board:
        {super().__str__()}
        """


    def gamenode_str(self):
        return super().__str__()


    def Q_value(self) -> float:
        """ Compute the Q value (average observed eval) of the node """

        return self.total_value / (1 + self.num_visits)


    def u_value(self) -> float:
        """ Compute the U value (decaying prior) of the node """

        siblings = 1
        
        if self.prev is not None:
            siblings = np.sqrt(self.prev.num_visits - 1) # Same as sum of siblings including self

        return C_PUCT * self.prior * siblings / (self.num_visits + 1)


    def create_child(self, loc) -> TreeNode:
        """
        Returns the child TreeNode after placing a stone at the
        location provided (NO ERROR CHECKING)

        Args:
            loc: index tuple for the location to place the stone
                 or (-1, -1) to pass
        """

        child = TreeNode(super().copy())
        child.play_stone(loc[0], loc[1], move = True) #it works ok

        child.prev = self
        child.prev_move = loc

        return child


    @staticmethod
    def lazy_child(parent: TreeNode, loc) -> TreeNode:
        """
        Returns a child TreeNode for the move at loc without creating its
        board, which is only created once needed (most children are never visited)

        Args:
            parent: the TreeNode to create the child of
            loc: index tuple for the location of the move
                 or (-1, -1) to pass
        """

        child = TreeNode.__new__(TreeNode)

        child.prev = parent
        child.prev_move = loc
        child.nexts = []

        child.num_visits = 0
        child.total_value = 0
        child.prior = 0

        return child


    def create_board(self) -> None:
        """ Creates the board of a lazy child by playing its move on a copy of the parent """

        board = GameNode.copy(self.prev)
        board.play_stone(self.prev_move[0], self.prev_move[1], move = True)

        # Keep tree data (prev, prev_move, nexts)
        for key, value in board.__dict__.items():
            self.__dict__.setdefault(key, value)


    def is_leaf(self) -> bool:
        """ Returns if this is a leaf OR is terminal """

        return (len(self.nexts) == 0) or self.is_terminal()


    def get_children(self, allow_pass: bool) -> List[Self]:
        """
        Returns a list of new lazy TreeNodes for each valid move
        from the current position

        Args:
            allow_pass: whether to consider pass as a valid move
        """

        self.nexts = [TreeNode.lazy_child(self, move) for move in self.available_moves()]

        # Pass is last, keep it if there are no other moves
        if not allow_pass and len(self.nexts) > 1:
            self.nexts.pop()

        return self.nexts

    
    def backprop(self, value):
        """
        Add value to each TreeNode's total_val and increment the num_visits 
        for each TreeNode node up until the root

        Args:
            value: The value computed by monte_carlo eval
        """
        self.num_visits += 1
        self.total_value += value

        if self.prev is not None:
            self.prev.backprop(value)


    def get_policy(self, temperature: float = None) -> torch.Tensor:
        """
        Calculates the MTCS policy given by the exponentiated visit count, ie:
        num_visits(action)^(1/temperature)/total_num_visits^(1/temperature)
        Ordered by nexts

        Args:
            temperature: the temperature to use (None to use the move schedule)
        """

        logits = np.array([child.num_visits for child in self.nexts], dtype=np.float64)

        # Fall back to uniform if no children were visited
        if logits.sum() == 0:
            logits = np.ones_like(logits)

        if temperature is None:
            temperature = 1 if self.move < NUM_MOVES_MAX_TEMPERATURE else .1

        logits = logits ** (1 / temperature)

        probs = logits / logits.sum()

        return torch.tensor(probs)

