from game_node import GameNode
from typing import List, Self
from __future__ import annotations


class TreeNode(GameNode):
    """
    GameNode wrapper for MCTS

    Args:
        gn: The GameNode to copy from
        num_visits: initial visit count
        total_value: initial total value
        prior: output from the policy head (should be const)
    """


    def __init__(self, gn: GameNode, num_visits = 0, total_value = 0, prior = 0):
        self.__dict__.update(gn.__dict__)

        self.num_visits = num_visits
        self.total_value = total_value
        self.prior = prior


    def Q_value(self) -> float:
        """ Compute the Q value (average observed eval) of the node """

        return self.total_value / (1 + self.num_visits)


    def u_value(self) -> float:
        """ Compute the U value (decaying prior) of the node """

        return self.prior / (self.num_visits + 1)


    def create_child(self, loc) -> TreeNode:
        """
        Returns the child TreeNode after placing a stone at the
        location provided (NO ERROR CHECKING)

        Args:
            loc: index tuple for the location to place the stone
                 or (-1, -1) to pass
        """

        child = self.copy()
        child = TreeNode(child)

        self.children.append(child)
        child.prev = self
        child.prev_move = loc

        return child


    def is_leaf(self) -> bool:
        """ Returns if this is a leaf OR is terminal """

        return (len(self.nexts) == 0) or self.is_terminal()


    def get_children(self, allow_pass: bool) -> List[Self]:
        """
        Returns a list of new TreeNodes for each valid move
        from the current position
    
        Args:
            allow_pass: whether to consider pass as a valid move
        """

        raise NotImplementedError()

    
    def backprop(value):
        """
        Add value to each TreeNode's total_val and increment the num_visits 
        for each TreeNode node up until the root

        Args:
            value: The value computed by monte_carlo eval
        """

        raise NotImplementedError()
