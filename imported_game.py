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
        Returns the head of a linked list made of
        GameNodes that represents the game. It is
        guaranteed len(node.nexts) <= 1 for all
        nodes in the linked list
        """

        head = GameNode(9)
        curr = head

        for move in self.moves:
            curr = curr.create_child(move)
        
        return head

    
    def __repr__(self):
        return f"ImportedGame({self.path})"


    def __str__(self):
        return self.__repr__()


if __name__ == "__main__":
    # Load a game
    GAME_PATH = "games/001001.sgf"

    game = ImportedGame(GAME_PATH)
    node = game.linked_list()

    while True:
        print(node)
        
        if len(node.nexts) == 1:
            node = node.nexts[0]
        else:
            break

