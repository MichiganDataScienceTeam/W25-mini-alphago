from typing import Self
from game_node import GameNode
from network import NeuralNetwork
from supervised_learning import GoLoss

class TreeNode(GameNode, GoLoss):

    #Initial State -- Prior already computed, Q is empty
    def __init__(self, num_visits = 0, total_val = 0, prior = 0):
        self.num_visits = num_visits
        self.total_val = total_val
        self.prior = prior
    
    #Create children! Need to convert children GameNodes to TreeNodes
    
    #Select step: start at current node, compute argmax(Q+ u)
    def select_best_node(self):
        for i in GameNode.nexts:
            best_node = GameNode.nexts[0]
            best_value = 0
            current_val = (GameNode.nexts[i].total_val/GameNode.nexts[i].num_visits + GameNode.nexts[i].prior/GameNode.nexts[i].num_visits)
            if(current_val > best_value):
                best_node = GameNode.nexts[i]
                best_value = current_val
        return best_node

    #Evaluate and Backup: start at selected node from prior step, eval pos, each eval counts as a visit, update
    def evaluate_node(self):
        ''
        for i in self.nexts:
            model = NeuralNetwork().float()
            GoLoss.train(model)
        #evaluate each child node of selected node
        #initialize p

    #Update all parents
    def backup(self):
        ''
        #update 
        while(self.prev):
            self.prev.num_visits += 1
            self.prev.total_val += self.total_val
            self = self.prev


    if __name__ == "__main__":
        n = 5
        for i in n:
            best_node = select_best_node(self)
            evaluate_node(best_node) 
            backup(best_node)
