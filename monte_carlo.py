from network import NeuralNet
from tree_node import TreeNode
import numpy as np

from numpy.typing import NDArray
from typing import Tuple

from data_preprocess import node_to_tensor

from game_node import GameNode


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

        self.curr = root

    def __str__(self):
        # return f"""Current node: {str(self.curr)}
        # Current node children: {'[' + ', '.join([str(a) for a in self.curr.nexts]) + ']'}
        # """
        return f"""Current node: {str(self.curr)}
        Current node children: {'[' + ', '.join([str(a) for a in self.things]) + ']'}
        """

    def select(self, node: TreeNode) -> TreeNode:
        """
        Returns the leaf node from the subtree rooted at node
        resulting from greedily maximizing Q+U for all children

        Args:
            node: the TreeNode to start selection from
        """

        while not node.is_leaf():
            # Select the child with the highest Q + U value
            best_child = max(node.nexts, key=lambda child: child.Q_value() + child.u_value())
            node = best_child

        return node
    
    
    def evaluate(self, node: TreeNode) -> Tuple[float, NDArray]:
        """
        Get the network's eval (value AND policy) of the current game state

        Consider evaluating on all d8 transformations and taking the mean
        if model is fast enough

        Args:
            node: the node to evaluate
        """

        out = self.model.forward(node_to_tensor(node).unsqueeze(0))
        return out[1].item(), out[0].squeeze(0).detach().numpy()
    

    def expand(self, node: TreeNode, prior: NDArray, allow_pass: bool = True) -> None:
        """
        Adds all valid children to the tree an initializes
        all values as described in the slides

        Args:
            node: the TreeNode from select
            prior: the precomputed output from the policy head
        """

        node.get_children(allow_pass=allow_pass)
        for i, child in enumerate(node.nexts):
            child.num_visits = 0
            child.total_value = 0
            child.prior = prior[i]

    
    def choose_action(self, temperature: float, node: TreeNode) -> Tuple[int, int]:
        """
        Samples an action from the distribution given by the exponentiated visit count, ie:
        num_visits(action)^(1/temperature)/total_num_visits^(1/temperature)

        Note that this must use some random function and is never purely deterministic

        Args:
            temperature: Hyperparameter from (0, 1] that selects for how much exploration you want the model to perform 
            (higher more exploration, lower less)
        """

        denom = sum(child.num_visits ** (1 / temperature) for child in node.nexts)
        probs = [child.num_visits ** (1 / temperature) / denom for child in node.nexts]
        action_index = np.random.choice(len(node.nexts), p=probs)

        return node.nexts[action_index].prev_move
    

    # TEMP
    def search(self) -> None:
        selected = self.select(self.curr)
        val, policy = self.evaluate(selected)
        self.expand(selected, policy, allow_pass=False)
        selected.backprop(val)
        self.things.append(selected)
        print(f"Selected node after search:\n{selected}")


if __name__ == "__main__":

    nn = NeuralNet()
    root_node = GameNode(9)
    
    game_tree = MonteCarlo(
        model=nn,
        root=TreeNode(root_node)
    )

    game_tree.things = []

    game_tree.search()
    game_tree.search()
    game_tree.search()
    print(game_tree)


    # val, priors = game_tree.evaluate(root_node)
    # game_tree.expand(root_node, priors)

    # chosen_node = 
    
    