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
                 nexts: list[Self] = None):

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
        if nexts is not None:
            self.nexts = nexts
        else:
            self.nexts = []


    def copy(self) -> Self:
        """Returns a deep copy of this GameNode"""

        res = GameNode(
            size = self.size,
            komi = self.komi,
            move = self.move,
            prev = self.prev,
            prev_move = self.prev_move,
            nexts = []
        )

        # Board deep copy
        res.grid = self.grid.copy()
        res.seen = self.seen.copy()

        res.num_passes = self.num_passes

        res.groups = [group.copy() for group in self.groups]

        res.index = Board.board_index
        Board.board_index += 1

        return res


    def is_terminal(self) -> bool:
        # First, check if the game is terminal by other means.
        if super().is_terminal():
            return True
        
        # Check for double pass: if this node's move is a pass
        # and the parent's move is also a pass.
        if self.prev is not None:
            # Make sure the parent's prev_move is defined before checking
            if self.prev.prev_move == (-1, -1) and self.prev_move == (-1, -1):
                return True

        return False


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

        child.prev = self
        child.prev_move = loc

        return child


if __name__ == "__main__":
    board = GameNode(size = 9)

    while not board.is_terminal():
        try:
            print("\nSelect a move")
            row = int(input("Row: "))
            col = int(input("Column: "))

            next_board = board.create_child((row, col))
            board.nexts.append(next_board)
            board = next_board

            print(board.get_game_data())
            print(board.get_game_data().shape)

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
