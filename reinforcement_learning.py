from bot import MonteCarloBot
import json



def self_play(bot: MonteCarloBot, num_moves: int = 250, verbose: bool = False):
    """
    Plays a game of self-play using the bot.

    num_moves: The number of moves to cap at
    """

    # Log the moves and policies
    moves = []
    policies = []

    # Play the game
    while not bot.mcts.curr.is_terminal() and bot.mcts.curr.move < num_moves:
        move = bot.choose_move()
        bot.make_move(move)

        moves.append(move)

        # Policy is 82 len array, policy[0] is moving on top left corner, policy[81] is passing, etc.
        BOARD_SIZE = bot.mcts.curr.size
        policy = [0] * ((BOARD_SIZE * BOARD_SIZE) + 1)
        probs = bot.mcts.get_policy(1.0, bot.mcts.curr)

        for i, child in enumerate(bot.mcts.curr.nexts):
            prev_move = child.prev_move
            if prev_move == (-1, -1):
                policy[BOARD_SIZE * BOARD_SIZE] = probs[i]
            else:
                policy[prev_move[0] * BOARD_SIZE + prev_move[1]] = probs[i]
        policies.append(policy)

        if verbose:
            print(f"Move {bot.mcts.curr.move}: {move}")
            print(bot.mcts.curr)

    return {'moves': moves, 'policies': policies, 'winner': bot.mcts.curr.compute_winner()}



if __name__ == "__main__":
    bot = MonteCarloBot()
    game_data = self_play(bot, verbose=True)
    with open("game_data.json", "w") as f:
        json.dump(game_data, f)