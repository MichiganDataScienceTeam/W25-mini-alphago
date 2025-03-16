from game_node import GameNode
from typing import Any

class ImportedGame:
    """
    Wrapper for easy handling of .sgf files describing
    9x9 Go games.

    Args:
        file_path: path to .sgf file
    """

    
    def __init__(self, file_path: str):
        # Save file path
        self.path = file_path

        # Parsing helper
        split_args = lambda s: {x.split("[")[0]: x.split("[")[1] for x in s.split("]") if x}

        with open(file_path, "r") as f:
            raw = f.read().strip()

            # Remove leading and trailing parentheses
            raw = raw.lstrip("(")
            raw = raw.rstrip(")")

            # Remove all newlines
            raw = raw.replace("\n", "")

            # Separate metadata and game data
            temp = raw.split(";")
            meta = split_args(temp[1])
            game = [split_args(x) for x in temp[2:]]

            # Save meta
            self.meta = {
                "size": int(meta["SZ"]),
                "komi": float(meta["KM"]),
                "winner": {"B": 1, "W": 2}[meta["RE"][0]],
                "final_eval": {"B": 1, "W": -1}[meta["RE"][0]],
                "player 1": meta["PB"],
                "player 2": meta["PW"]
            }

            # Generate move list
            moves = []

            turn = 1
            turn_to_color = [None, "B", "W"]

            for move in game:
                color = turn_to_color[turn]
                col = ord(move[color][0]) - ord("a")
                row = ord(move[color][1]) - ord("a")

                if col >= self.meta["size"] or row >= self.meta["size"]:
                    col = -1
                    row = -1
                else:
                    moves.append((row, col))

                turn = 1 if turn == 2 else 2
            
            # Save data
            self.moves = moves


    def linked_list(self) -> GameNode:
        """
        Returns the tail of a linked list made of
        GameNodes that represents the game. It is
        guaranteed len(node.nexts) <= 1 for all
        nodes in the linked list
        """

        curr = GameNode(9)

        for move in self.moves:
            curr = curr.create_child(move)
        
        return curr


    def d4_transformations(self) -> list[list[tuple[int, int]]]:
        """
        Returns a list of all 8 transformations in the
        dihedral group of a square applied to the move
        list (includes the identity)
        """

        size = self.meta["size"]

        rotate_90_clockwise = lambda moves: [(move[1], size - move[0] - 1) for move in moves]
        vertical_flip = lambda moves: [(size - move[0] - 1, move[1]) for move in moves]

        I = self.moves

        R1 = rotate_90_clockwise(I)
        R2 = rotate_90_clockwise(R1)
        R3 = rotate_90_clockwise(R2)

        F1 = vertical_flip(I)
        F2 = vertical_flip(R2)

        D1 = vertical_flip(R1)
        D2 = vertical_flip(R3)

        return [I, R1, R2, R3, F1, F2, D1, D2]

    
    def __repr__(self):
        return f"ImportedGame({self.path})"


    def __str__(self):
        return self.__repr__()


if __name__ == "__main__":
    # Load a game
    GAME_PATH = "games/001001.sgf"

    game = ImportedGame(GAME_PATH)
    node = game.linked_list()

    # Move to head
    while node.prev is not None:
        node = node.prev

    while True:
        print(node)
        
        if len(node.nexts) == 1:
            node = node.nexts[0]
        else:
            break

