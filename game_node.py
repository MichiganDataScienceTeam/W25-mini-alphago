import numpy as np
from typing import Self

from board import Board

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


    def create_child(self, loc: tuple[int, int]) -> Self:
        """
        Returns the child GameNode after placing a stone at the
        location provided

        Args:
            loc: index tuple for the location to place the stone
                 or (-1, -1) to pass
        """

        child = self.copy()

        if not child.play_stone(loc[0], loc[1], True):
            raise ValueError(f"Invalid move location \"{loc}\"")

        self.nexts.append(child)
        child.prev = self
        child.prev_move = loc

        return child


    def get_game_data(self, history_length=3) -> np.array:
        """
        Returns a numpy array of white and black board states + past history_length moves
        """

        game_data = []

        node = self  # through node history
        for i in range(history_length):
            if node is None:
                empty_board = np.zeros((2, self.size, self.size), dtype=int)
                game_data.append(empty_board)
                continue

            white_grid = (node.grid == 2).astype(int)
            black_grid = (node.grid == 1).astype(int)

            game_data.append(np.array([white_grid, black_grid]))
            node = node.prev

        # make an arbitrary array with the length equal to the number of moves made
        num_moves = 0
        n = self
        while n.prev is not None:
            num_moves += 1
            n = n.prev
        game_data.append(np.array([1] * num_moves))

        return game_data


if __name__ == "__main__":
    board = GameNode(size = 9)

    while not board.is_terminal():
        try:
            print("\nSelect a move")
            row = int(input("Row: "))
            col = int(input("Column: "))

            board = board.create_child((row, col))
            print(board.get_game_data())

        except KeyboardInterrupt:
            print("\nKeyboard Interrupt. Game Ended")
            break
        # except:
        #     print("Error while processing move. Try again.")
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
