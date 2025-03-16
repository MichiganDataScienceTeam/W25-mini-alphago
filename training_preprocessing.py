import numpy as np
from board import Board
from game_node import GameNode
from imported_game import ImportedGame
import torch.nn as nn


class Dataset:
    #takes in a final gamenode
    #makes tuples of s,z,pi
    #implement loss function
    def createTuples(ImportedGame):
        GAME_PATH = "games/001001.sgf"
        game = ImportedGame(GAME_PATH)    
        s = nn.tensor(ImportedGame.linked_list(), 9,9)
        z = ImportedGame.meta["final_eval"]
        pi = nn.tensor(game+1,)

    def lossFunction():

