from network import NeuralNet
from tree_node import TreeNode

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

        out = self.model.forward(node_to_tensor(node).unsqueeze(0))
        return out[1].item(), out[0].squeeze(0).detach().cpu().numpy() # trust me guys it works
    

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

    
    def get_policy(self, temperature: float, node: TreeNode) -> NDArray:
        """
        Calculates the MTCS policy given by the exponentiated visit count, ie:
        num_visits(action)^(1/temperature)/total_num_visits^(1/temperature)

        Args:
            temperature: Hyperparameter from (0, 1] that selects for how much exploration you want the model to perform 
            (higher more exploration, lower less)
        """

        denom = sum(child.num_visits ** (1 / temperature) for child in node.nexts) + len(node.nexts)
        probs = [(child.num_visits ** (1 / temperature) + 1) / denom for child in node.nexts]

        return probs
    

    def search(self) -> None:
        """
        Performs one iteration of search

        Note that this doesn't return a value because the goal of search is improve the tree,
        not to actually "search" for a specific node.
        """

        selected = self.select(self.curr)
        val, policy = self.evaluate(selected)
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
    
    