import numpy as np
from board import Board
from game_node import GameNode

b = GameNode(9)

b = b.create_child([1,3])
b = b.create_child([3,2])
b = b.create_child([4,4])
b = b.create_child([3,3])
b = b.create_child([0,1])
b = b.create_child([1,0])
b = b.create_child([2,1])
b = b.create_child([2,2])

class preprocessing:
    #takes in a game node, updates board, then repeats with previous turn adding to history each time
    #not sure if boards are in the correct order
    def createHistory(gameNode) -> np.array:
        white_board = np.zeros((9,9))
        black_board = np.zeros((9,9))
        history = np.zeros((5, 9, 9))
        if gameNode.move % 2 == 0:
            black_board[gameNode.prev_move[0]][gameNode.prev_move[1]] = 1
            history[4] = black_board
        else:
            white_board[gameNode.prev_move[0]][gameNode.prev_move[1]] = 1
            history[4] = white_board
        gameNode = gameNode.prev
        for i in range(2):
            if gameNode.move % 2 == 0:
                black_board[gameNode.prev_move[0]][gameNode.prev_move[1]] = 1
            else:
                white_board[gameNode.prev_move[0]][gameNode.prev_move[1]] = 1
            history[i*2] = black_board
            history[1+i*2] = white_board
            gameNode = gameNode.prev
        print(history)
        return history
preprocessing.createHistory(b)
