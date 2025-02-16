from board import *
from game_node import *
import numpy as np


def current_turn_mat(node: GameNode):
    size = node.size
    black_grid = np.array((size, size), dtype=int)
    white_grid = np.array((size, size), dtype=int)

    black_grid = (node.grid == 1).astype(int)
    white_grid = (node.grid == 2).astype(int)

    #g = np.array([black_grid, white_grid])
    return black_grid, white_grid

def traverse_prev(node: GameNode):
    stacked_grid = []
    if node.move % 2 == 0:
        stacked_grid.append(np.full((node.size, node.size), 1))
    else:
        stacked_grid.append(np.full((node.size, node.size), 2))
    limit = 4
    visit_node = node

    while limit > 0 and visit_node is not None:
        b, w = current_turn_mat(visit_node)
        stacked_grid.append(b)
        stacked_grid.append(w)
        visit_node = visit_node.prev
        limit -= 1

    stacked_grid = np.array(stacked_grid)

    return stacked_grid
    
def play_game():
    board = GameNode(size = 9)

    for i in range(2):
        try:
            print("\nSelect a move")
            row = int(input("Row: "))
            col = int(input("Column: "))

            board = board.create_child((row, col))
        except KeyboardInterrupt:
            print("\nKeyboard Interrupt. Game Ended")
            break
        except:
            print("Error while processing move. Try again.")
        else:
            print(board)

    scores = board.compute_simple_area_score()

    print("Stats:")
    print(f"Player 1 (black) score: {scores[0]}")
    print(f"Player 2 (white) score: {scores[1]}")
    print(f"Final score eval: {board.compute_winner()}")

    print("PATH:")

    return board

    