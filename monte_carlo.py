from network import NeuralNet
from tree_node import TreeNode

from numpy.typing import NDArray
from typing import Tuple

from data_preprocess import node_to_tensor

from game_node import GameNode

from config import *

class MonteCarlo:
    """
    Monte Carlo Tree Search

    Wrap Tree Node w/ Evaluation Function

    Args:
        model: the NN to use
    """


    def __init__(self, model: NeuralNet, root: TreeNode, device: str = DEVICE):
        self.device = device
        self.model = model.to(self.device)
        self.root = root
        self.curr = root

    def __str__(self):
        return f"""Current node: {str(self.curr)}
        Current node children: {'[' + ', '.join([str(a) for a in self.curr.nexts]) + ']'}
        """

    def select(self, node: TreeNode) -> TreeNode:
        """
        Returns the leaf node from the subtree rooted at node
        resulting from greedily maximizing Q+U for all children

        Args:
            node: the TreeNode to start selection from
        """

        while not node.is_leaf():
            # Select the child with the best Q + U value
            coef = 1 if (node.move % 2) == 0 else -1
            best_child = max(node.nexts, key=lambda child: coef * child.Q_value() + child.u_value())
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

        self.model.eval()
        with torch.no_grad():
            out = self.model.forward(node_to_tensor(node).unsqueeze(0).to(self.device))
        return out[1].item(), out[0].squeeze(0).detach().cpu().numpy() # trust me guys it works
    

    def expand(self, node: TreeNode, prior: NDArray, allow_pass: bool = True) -> None:
        """
        Adds all valid children to the tree an initializes
        all values as described in the slides

        Args:
            node: the TreeNode from select
            prior: the precomputed output from the policy head
        """
        if node.is_terminal():
            return

        node.get_children(allow_pass=allow_pass)
        for i, child in enumerate(node.nexts):
            if i == 82:
                print(len(set(node.nexts)))
                print(len(node.nexts))
            child.num_visits = 0
            child.total_value = 0
            child.prior = prior[i]
    

    def search(self) -> None:
        """
        Performs one iteration of search

        Note that this doesn't return a value because the goal of search is improve the tree,
        not to actually "search" for a specific node.
        """

        selected = self.select(self.curr)
        val, policy = self.evaluate(selected)
        if (self.curr.move > NUM_MOVES_ALLOW_PASS):
            self.expand(selected, policy, allow_pass=True)
        else: 
            self.expand(selected, policy, allow_pass=False)
        selected.backprop(val)
    

    def move_curr(self, loc: Tuple[int, int]) -> None:
        """
        Moves the curr node forward by playing the action at loc

        Args:
            loc: the location of the move to play
        """

        for child in self.curr.nexts:
            if child.prev_move == loc:
                self.curr = child
                return

        raise ValueError(f"Child from move at {loc} not found. Maybe you forgot to search?")


if __name__ == "__main__":
    nn = NeuralNet()
    root_node = GameNode(9)
    
    game_tree = MonteCarlo(
        model=nn,
        root=TreeNode(root_node)
    )

    game_tree.search()

    print(game_tree)
    
    