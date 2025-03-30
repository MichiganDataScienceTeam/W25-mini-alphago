# Mini-AlphaGo, Winter 2025

In this project, we will build a modified hybrid of [AlphaGo](https://www.nature.com/articles/nature16961) and [AlphaZero](https://www.nature.com/articles/nature24270) for 9x9 Go from almost scratch (using only computational libraries).

## Schedule

| **Week** | **Links** |
| --- | --- |
| 1 | [Slides](https://docs.google.com/presentation/d/1-xUB_iLC-hbhHI7JJtxdNb0yJfJjoJYizdwDDreBi8k/edit?usp=sharing), [Play Online Go](https://online-go.com/), [Computer Go Rules](https://tromp.github.io/go.html)
| 2 | [Slides](https://docs.google.com/presentation/d/1Tl5gFVL9Pp-qJr6oYB78062bHCQMhRM6sasR86lCWHc/edit?usp=sharing), [NN Arch](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf#page=27), [PyTorch CNN example](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) |
| 3 | [Slides](https://docs.google.com/presentation/d/1yF0llAtNVfPCPmMXlIslRqWAPUI-A_-ez7vHlBAoXFc/edit?usp=sharing), [PyTorch Modules](https://pytorch.org/docs/stable/notes/modules.html#modules-as-building-blocks) |
| 4 | [Slides](https://docs.google.com/presentation/d/1kch18ub-a1Mqck-qaXYb6X_Oq3d1L4-R20YT2aCWkdE/edit?usp=sharing), [SL Description](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf#page=25) |
| 5 | [Slides](https://docs.google.com/presentation/d/1LEbv5XmeUX-hytKAuPFUy0yyKeiUiF1bOkyMBQJuSTI/edit?usp=sharing) |
| 6 | [Slides](https://docs.google.com/presentation/d/1TNW__iBweAtqnh088tEdCIuehKC8lB6dVtbhaNIpi_8/edit?usp=sharing), [AGZ MCTS Algorithm](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf#page=25) |
| 7 | [Slides](https://docs.google.com/presentation/d/1wIUXCe9MaicW_u0sWpuu7ez-zFpV4AijGfOX_aeQJbA/edit?usp=sharing), [AGZ MCTS Algorithm](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf#page=25) |

For a more detailed list of topics and resources, check the most recent "This Week in Mini-AlphaGo" email (released every Wednesday afternoon).

## General Usage

### `Board` and `GameNode`

`board.py` contains the `Board` class, which is a custom Go board interface written specifically for this project. `Board` handles all the game logic of Go, including moves, captures, end detection, and scoring. The file contains example usage and docstrings for each function.

It is important to note that `Board` objects contain a `.grid`, which is a `np.NDArray` with shape `(size, size)` containing `int`s that are `0` (empty), `1` (black stone), or `2` (white stone).

`GameNode` inherits from `Board` and implements a tree structure with `.prev` (the `GameNode` leading to `self`), `.prev_move` (the move tuple that goes from `.prev` to `self`), and `.nexts` (a possibly incomplete list of child `GameNode`s).

**Examples**

Count the number of black stones on a `Board` after playing some moves
```py
board = Board(9)  # Init empty 9x9 Board

board.play_stone(5, 6)  # Black plays at (5, 6)
board.play_stone(0, 1)  # White plays at (0, 1)
board.play_stone(0, 0)  # Black plays at (0, 0)

print(board)  # Visual board representation
print(f"Black stones: {(board.grid == 1).sum()}")  # 2
```

Print history leading up to a `GameNode`
```py
node = GameNode(9)  # Init empty 9x9 GameNode

node = node.create_child((5, 6))  # Black plays at (5, 6)
node = node.create_child((0, 1))  # White plays at (0, 1)
node = node.create_child((0, 0))  # Black plays at (0, 0)

# Print game history in reverse
while node != None:
    print(node)
    node = node.prev
```

### Misc

`elo_calculator.py`, contains the `Elo_calculator` class, which allows you to play to `Bot`s against each other, calculate their updated elos, and store them.

`bot.py` has the basic structure for the `Bot` class, which will be an abstraction for the Go bots we will build that provides a consistent interface for evaluation.

In the following code:
```python
go = Board(9)
elo = Elo_calculator(game=go, prev_elo_data = "elo_data.pkl")

random_player1 = Bot()
random_player2 = Bot()

elo.register_bot(name= "bob", bot = random_player1)
elo.register_bot(name = "bobette", bot = random_player2)

elo.play_match("bob", "bobette")

elo.save("elo_data.pkl")
```
We create an elo_calculator object from past elo data, register 2 agents, and run them against each other, saving the data to 'elo_data.pkl'. If you were to run this code again, then the code below would do the same this as the code snippet above:
```python
go = Board(9)
elo = Elo_calculator(game=go, prev_elo_data = "elo_data.pkl")

elo.play_match("bob", "bobette")
elo.save("elo_data.pkl")
```
