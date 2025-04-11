from __future__ import annotations
from game_node import GameNode
from typing import List, Self


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

        return self.prior / (self.num_visits + 1)


    def create_child(self, loc) -> TreeNode:
        """
        Returns the child TreeNode after placing a stone at the
        location provided (NO ERROR CHECKING)

        Args:
            loc: index tuple for the location to place the stone
                 or (-1, -1) to pass
        """

        child = super().copy()
        super(type(child), child).play_stone(loc[0], loc[1], move = True) #it works ok
        child = TreeNode(child)

        child.nexts = []
        self.nexts.append(child)
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
        children = []
        list_of_moves = self.available_moves()
        for move in list_of_moves:
            # If pass isn't a valid move, don't include in childen list
            if not allow_pass and move == (-1, -1):
                continue

            children.append(self.create_child(move))
            
        return children

    
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
