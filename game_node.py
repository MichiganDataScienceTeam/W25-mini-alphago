from typing import Self

from board import Board

import numpy as np

class GameNode(Board):
    """
    Internal class for GameTree

    Args:
        size: length of one dimension of the board
        komi: The komi applied to white's score
        move: move number at current position (default = 0)
        prev: parent GameNode
        prev_move: the move played at the parent to make this
        nexts: list of child GameNodes
    """


    def __init__(self, size: int, komi: float = 7.5, move: int = 0,
                 prev: Self = None, prev_move: tuple[int, int] = None,
                 nexts: list[Self] = []):
        
        if komi - int(komi) == 0:
            raise ValueError(f"Invalid komi {komi}: komi must contain" + 
                             " a fractional tie-breaker")
        
        super().__init__(
            size = size,
            komi = komi,
            move = move
        )

        self.prev = prev
        self.prev_move = prev_move
        self.nexts = nexts


    def copy(self) -> Self:
        """Returns a deep copy of this GameNode"""

        res = GameNode(
            size = self.size,
            komi = self.komi,
            move = self.move,
            prev = self.prev,
            prev_move = self.prev_move,
            nexts = self.nexts.copy()
        )

        # Board deep copy
        res.grid = self.grid.copy()
        res.seen = self.seen.copy()

        res.num_passes = self.num_passes

        res.groups = [group.copy() for group in self.groups]

        res.index = Board.board_index
        Board.board_index += 1

        return res


    def play_stone(self, row: int, col: int, move: bool) -> None:
        """
        GameNode shouldn't support play_stone because it violates
        tree invariants. create_child handles the tree components
        while maintaining the same general function
        """

        raise Exception("GameNode doesn't support play_stone. Use create_child instead.")


    def __bad_play_stone(self, row: int, col: int, move: bool) -> bool:
        """
        Board.play_stone passthrough for internal GameNode
        use only.
        """

        return super().play_stone(row, col, move)


    def create_child(self, loc: tuple[int, int]) -> Self:
        """
        Returns the child GameNode after placing a stone at the
        location provided

        Args:
            loc: index tuple for the location to place the stone
                 or (-1, -1) to pass
        """

        child = self.copy()

        if not child.__bad_play_stone(loc[0], loc[1], True):
            raise ValueError(f"Invalid move location \"{loc}\"")

        self.nexts.append(child)
        child.prev = self
        child.prev_move = loc

        return child

    def generate_history(self, moves_back: int):
        history = []
        node = self
        board_size = self.size
        black_move = (self.move % 2 == 0)
        
        for _ in range(moves_back):
            if node is None:
                white_grid = np.zeros((board_size, board_size), dtype=int)
                black_grid = np.zeros((board_size, board_size), dtype=int)
                history.append(white_grid)
                history.append(black_grid)
                continue
            
            white_grid = (node.grid == 2).astype(int)
            black_grid = (node.grid == 1).astype(int)

            history.append(white_grid)
            history.append(black_grid)
        
            node = node.prev
        
        np_history = np.array(history[::-1])
        new_board = np.full((board_size, board_size), -1 if black_move else 1, dtype=int)
        np_history = np.append(np_history, [new_board], axis=0)
        return np_history

# if __name__ == "__main__":
#     gn= GameNode(9)
#     gn = gn.create_child((1,1))   
#     gn = gn.create_child((1,2))
#     print(gn.generate_history(3))

if __name__ == "__main__":
    board = GameNode(size = 9)

    while not board.is_terminal():
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

    while board.prev != None:
        board = board.prev
    
    while len(board.nexts) != 0:
        print(board)
        board = board.nexts[0]