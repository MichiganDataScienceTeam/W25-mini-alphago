from board import *

board = Board(5)


while not board.is_full():
    print("Black's move: Type move in format [row] [col], or type 'pass' to pass")
    move = input()
    if move.lower() == "pass":
        board.play_stone(1, 0, 0, move=False)
    else:
        input_list = move.split(" ")
        if len(input_list) != 2:
            print("invalid move")
            exit(1)
        board.play_stone(1, int(input_list[0]), int(input_list[1]))
    
    print(board)
    
    if board.is_full(): break

    print("White's move: Type move in format [row] [col], or type 'pass' to pass")
    move = input()
    if move.lower() == "pass":
        board.play_stone(2, 0, 0, move=False)
    else:
        input_list = move.split(" ")
        if len(input_list) != 2:
            print("invalid move")
            exit(1)
        board.play_stone(2, int(input_list[0]), int(input_list[1]))
    
    print(board)