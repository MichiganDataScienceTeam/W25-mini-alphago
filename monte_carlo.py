from game_node import GameNode
import network

class MonteCarlo(GameNode):
    def __init__():
        # Network information?_(self):
        ...

    
    def __init__(self, gn: GameNode, num_visits = 0, total_value = 0, prior = 0):
        self.__dict__.update(gn.__dict__)

        self.num_visits = num_visits
        self.total_value = total_value
        self.prior = prior


    def Q_value(self):
        return self.total_value / self.num_visits if self.num_visits != 0 else 0

    def u_value(self):
        return self.prior / (self.num_visits + 1)

    def create_child(self, loc):  
        child = self.copy()

        if not super(type(child), child).play_stone(loc[0], loc[1], True):
            raise ValueError(f"Invalid move location \"{loc}\"")

        child = MonteCarlo(child)

        self.children.append(child)
        child.prev = self
        child.prev_move = loc

        return child

    def get_nodes(self):
        """ 
            Returns a list of candidate leaf node(s)
        
            Runs the model on the current gamenode, returns n best nodes (moves)
                1. Take the first part of the tuple from calling forward on model
                2. Iterate through grid to find the n best moves
        """

    def evaluate_node():
        """
            Move to MonteCarlo
        """
    
    def backprop():        
        """"""
    
