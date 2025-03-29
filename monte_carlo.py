from network import NeuralNet
from tree_node import TreeNode
import numpy as np

from numpy.typing import NDArray
from typing import Tuple


class MonteCarlo:
    """
    Monte Carlo Tree Search

    Wrap Tree Node w/ Evaluation Function

    Args:
        model: the NN to use
    """


    def __init__(self, model: NeuralNet, root: TreeNode):
        self.model = model
        self.root = root


    def select(self, node: TreeNode) -> TreeNode:
        """
        Returns the leaf node from the subtree rooted at node
        resulting from greedily maximizing Q+U for all children

        Args:
            node: the TreeNode to start selection from
        """

        raise NotImplementedError()
    
    
    def evaluate(self, node: TreeNode) -> Tuple[float, NDArray]:
        """
        Get the network's eval (value AND policy) of the current game state

        Consider evaluating on all d8 transformations and taking the mean
        if model is fast enough

        Args:
            node: the node to evaluate
        """

        raise NotImplementedError()
    

    def expand(self, node: TreeNode, prior: NDArray) -> None:
        """
        Adds all valid children to the tree an initializes
        all values as described in the slides

        Args:
            node: the TreeNode from select
            prior: the precomputed output from the policy head
        """

        raise NotImplementedError()

    
    def choose_action(self, temperature: float):
        """
        Samples an action from the distribution given by the exponentiated visit count, ie:
        num_visits(action)^(1/temperature)/total_num_visits^(1/temperature)

        Note that this must use some random function and is never purely deterministic

        Args:
            temperature: Hyperparameter from (0, 1] that selects for how much exploration you want the model to perform 
            (higher more exploration, lower less)
        """

        raise NotImplementedError()

