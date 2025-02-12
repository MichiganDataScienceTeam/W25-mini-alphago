import numpy as np
from numpy.typing import NDArray

from typing import Self

from group import Group

class Board:
    """
    Go board with moves and basic point calculations

    Args:
        size: length of one dimension of the board
        komi: The komi applied to white's score
        move: move number at current position (default = 0)
    """


    board_index = 0


    def __init__(self, size: int, komi: float = 7.5, move: int = 0) -> None:
        if type(size) != int:
            raise TypeError("size must be an int")

        if not (5 <= size <= 19):
            raise ValueError("size must be in [5, 19]")
        
        self.size: int = size
        self.move: int = move
        self.seen: set[int] = set()
        self.groups: list[Group] = []

        self.grid = np.zeros((size, size), dtype=int)
        self.num_passes = 0
        self.komi = komi

        self.index = Board.board_index
        Board.board_index += 1
    

    @staticmethod
    def grid_to_int(grid: NDArray, turn: int):
        """
        Base 3 representation: 0 <= int(self) <= 3**(size**2)
        
        Args:
            grid: Board position represented as an numpy array
            turn: value of the stone to move (1 or 2)
        """

        return turn + ((3**np.arange(1, grid.shape[0]**2+1, dtype=object)) * grid.flatten()).sum()


    def __int__(self) -> int:
        """Base 3 representation: 0 <= int(self) <= 3**(size**2)"""

        return Board.grid_to_int(self.grid, self.move % 2 + 1)


    def __float__(self) -> float:
        """Enforce int over float"""

        raise TypeError("Board objects cannot be represented as a float")
    

    def __bool__(self) -> bool:
        """False iff empty"""

        return self.grid.any()


    def __str__(self) -> str:
        """
        Returns board with the header: 'Board {index} (NxN)'
        and 'Move #{move number}'
        and the NxN grid aligned and composed of
        of ┼, ├, ┤, ┬, ┴, ┌, ┐, └, ┘, ●, ○ with axis labels
        """

        out = []

        # Main Headers
        header1 = f"Board {self.index} ({self.size}x{self.size})"
        header2 = f"Move #{self.move}"

        out.append("   " + " "*((self.size * 2 - 1 - len(header1))//2) + header1)
        out.append("   " + " "*((self.size * 2 - 1 - len(header2))//2) + header2)

        # Column labels
        out.append("   " + "".join([f"{i} " for i in range(self.size)]))

        # Main grid
        for i in range(self.size):
            row = str(i).rjust(2) + " "
            for j in range(self.size):
                # Add spacing
                if j > 0:
                    row += " "

                # Stones
                if self.grid[i, j] == 1:
                    row += "○"
                    continue
                elif self.grid[i, j] == 2:
                    row += "●"
                    continue
                
                # Corners
                if i == 0 and j == 0:
                    row += "┌"
                elif i == 0 and j == self.size - 1:
                    row += "┐"
                elif i == self.size - 1 and j == 0:
                    row += "└"
                elif i == self.size - 1 and j == self.size - 1:
                    row += "┘"
                
                # Edges
                elif i == 0:
                    row += "┬"
                elif i == self.size - 1:
                    row += "┴"
                elif j == 0:
                    row += "├"
                elif j == self.size - 1:
                    row += "┤"
                
                # Middle
                else:
                    row += "┼"
            
            out.append(row)

        return "\n".join(out)

    
    def copy(self) -> Self:
        """Returns a deep copy of this Board"""

        res = Board(
            size = self.size,
            komi = self.komi,
            move = self.move
        )

        res.grid = self.grid.copy()
        res.seen = self.seen.copy()

        res.num_passes = self.num_passes

        res.groups = [group.copy() for group in self.groups]

        res.index = Board.board_index
        Board.board_index += 1

        return res
    

    def __repr__(self) -> str:
        """Returns Board constructor to a copy as a string (NON-EVALABLE!!!)"""

        return f"Board({self.size}x{self.size}, Move {self.move}, {int(self)})"


    # TODO: Fix very slow algorithm maybe
    # TODO: Choose better scoring method maybe
    def compute_simple_area_score(self) -> tuple[int, int]:
        """
        Returns a tuple of each player's score without komi:
            (player 1 score, player 2 score)

        Computed using the Area Scoring method:
            https://en.wikipedia.org/wiki/Rules_of_Go#Area_scoring
        I think this is equivalent to Tromp-Taylor scoring:
            https://tromp.github.io/go.html
        """

        if self.is_empty():
            return (0, 0)

        # Set up empty and filled
        empty = set(range(self.size**2))
        filled = set()

        for group in self.groups:
            filled |= group.intersections
        
        empty -= filled

        # Boundary expansion
        def get_boundary_single(idx: int) -> set[int]:
            """Returns boundary of location idx"""

            out = set()

            row = idx // self.size
            col = idx % self.size

            for i, j in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                # Prevent out of bounds
                if not (0 <= row + i < self.size):
                    continue

                if not (0 <= col + j < self.size):
                    continue
                    
                out.add(self.size * (row + i) + col + j)
            
            return out
        
        def get_boundary(locs: set[int]) -> set[int]:
            """Returns boundary of a set of locations"""

            out = set()

            for x in locs:
                out |= get_boundary_single(x)

            return out - locs

        def expand(start: int) -> tuple[set[int], set[int]]:
            """
            Returns interior and boundary of the maximal empty group
            containing start with bfs
            """

            seen = {start}
            prev_boundary = set()
            boundary = set()

            while True:
                boundary = get_boundary(seen)
                seen |= boundary & empty

                if boundary == prev_boundary:
                    break
                
                prev_boundary = boundary

            if not (boundary <= filled):
                raise Exception("WTF >:(")

            return seen, boundary
        
        def pick_one(s: set[int]) -> int:
            """Picks one element of s"""

            for x in s:
                return x
            
            raise ValueError("Cannot pick one from an empty set")

        # Count points
        flat_grid = self.grid.flatten()
        score = [0, 0]

        # Stone points
        for x in filled:
            score[flat_grid[x] - 1] += 1

        # Internal points
        used = set()

        while True:
            seen, boundary = expand(pick_one(empty - used))

            used |= seen

            candidate = flat_grid[pick_one(boundary)]

            for idx in boundary:
                if flat_grid[idx] != candidate:
                    score[candidate - 1] -= len(seen)
                    break
            
            score[candidate - 1] += len(seen)

            if used == empty:
                break
        
        return (score[0], score[1])


    def is_valid_move(self, row: int, col: int) -> bool:
        # Compute stone value
        val = self.move % 2 + 1

        # Pass is always allowed
        if (row, col) == (-1, -1):
            return True
        
        # On top of another stone is never allowed
        if self.grid[row, col] != 0:
            return False
        
        # If capturing, needs play_stone
        for group in self.groups:
            if group.group_type == val:
                continue
            
            if len(group - {(row, col)}) == 0:
                return self.__play_stone(row, col, False)
        
        # Prohibit suicide
        for group in self.groups:
            if group.group_type == val:
                continue

            if len(group - {(row, col)}) == 0:
                return False


    def play_stone(self, row: int, col: int, move: bool = True) -> bool:
        """
        Attempts to place a stone of value val at (row, col)
        
        Returns True if the move is valid, False if not

        Args:
            row: index of the row to place the stone
            col: index of the column to place the stone
            move (optional): whether or not to update the board, default True
        """

        return self.__play_stone(row, col, move)


    def __play_stone(self, row: int, col: int, move: bool = True) -> bool:
        """
        THIS IS A PRIVATE METHOD! DO NOT USE THIS OUTSIDE BOARD.PY

        Attempts to place a stone of value val at (row, col)
        
        Returns True if the move is valid, False if not

        Args:
            row: index of the row to place the stone
            col: index of the column to place the stone
            move (optional): whether or not to update the board, default True
        """

        # Compute stone value
        val = self.move % 2 + 1

        # Handle pass
        if (row, col) == (-1, -1):
            if move:
                self.num_passes += 1
                self.move += 1
            
            return True
        else:
            self.num_passes = 0

        # Prohibit placing stone on top of another
        if self.grid[row, col] != 0:
            return False
        
        # Create candidate grid after move is played
        candidate = self.grid.copy()
        candidate[row, col] = val

        # Find borders and liberties for new stone
        borders = set()
        liberties = set()

        for i, j in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            # Prevent out of bounds
            if not (0 <= row + i < self.size):
                continue

            if not (0 <= col + j < self.size):
                continue

            # Compute index
            idx = self.size * (row + i) + (col + j)

            # All adjacent are candidate borders
            borders.add(idx)

            # All empty borders are candidate liberties
            if self.grid[row + i, col + j] == 0:
                liberties.add(idx)
        
        # Create new stone group
        new_stone_group = Group(intersections = {self.size * row + col},
                                borders = borders,
                                liberties = liberties,
                                group_type = val)
        
        # Remove excess liberties
        new_stone_group.trim_liberties(self.groups)
        
        # Consolidate groups
        candidate_groups = Group.add_union(self.groups, new_stone_group)

        # Compute captures
        new_candidate_groups = []

        captured = set()

        for group in candidate_groups:
            # Skip same color
            if group.group_type == val:
                continue

            # Add if there are still liberties
            if len(group.liberties) > 0:
                new_candidate_groups.append(group)
                continue
            
            # Record captures
            captured |= group.intersections

        # Update for newly opened intersections
        for group in candidate_groups:
            group.replenish_liberties(captured)

        # Prohibit suicide
        for group in candidate_groups:
            # Skip opposite color
            if group.group_type != val:
                continue

            if len(group.liberties) == 0:
                return False
            
            new_candidate_groups.append(group)

        # Remove captured stones from the board
        for i in captured:
            candidate[i // self.size, i % self.size] = 0

        candidate_groups = new_candidate_groups

        # Prohibit repetition
        n = Board.grid_to_int(candidate, val)
        if n in self.seen:
            return False
        
        # Return allowed move immediately if necessary
        if not move:
            return True

        # Allow move
        self.grid = candidate
        self.groups = candidate_groups
        self.move += 1
        self.seen.add(n)

        return True


    def is_empty(self) -> bool:
        """Returns if the board is empty"""

        return np.all(self.grid == 0)


    def is_full(self) -> bool:
        """Returns if the board is full"""

        return np.all(self.grid != 0)


    def available_moves(self) -> list[tuple[int, int]]:
        """
        Returns a list of tuples corresponding to the row and
        column indices of available moves

        The tuple (-1, -1) corresponds with the move to pass
        """

        out = [(-1, -1)] # Players can always pass

        for i in range(self.size):
            for j in range(self.size):
                available = self.is_valid_move(
                    row = i,
                    col = j
                )

                if available:
                    out.append((i, j))
        
        return out


    def is_terminal(self) -> bool:
        """
        Returns if this is a terminal node (no possible children)
        """

        # Double pass ends the game
        if self.num_passes == 2:
            return True

        # No possible moves ends the game
        if self.is_full():
            return True
        
        # "Mercy rule"
        if self.move > 30 and len(self.groups) <= 1:
            return True

        # TODO: Are there more cases for terminal nodes?

        return False


    def compute_winner(self) -> int:
        """
        Returns 1 if player 1 wins and -1 if player 2 wins
        """

        p1_score, p2_score = self.compute_simple_area_score()

        p2_score += self.komi

        if p1_score > p2_score:
            return 1
        elif p1_score < p2_score:
            return -1
        
        raise ValueError(f"Failed to find winner with komi {self.komi}")


# Sample Code
if __name__ == "__main__":
    board = Board(size = 9)

    while not board.is_terminal():
        try:
            print("\nSelect a move")
            row = int(input("Row: "))
            col = int(input("Column: "))

            board.play_stone(row, col, True)
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

