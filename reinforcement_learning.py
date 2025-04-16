import torch

from tqdm import tqdm

from bot import MonteCarloBot
from dataset import Dataset
from network import GoLoss

from config import *


def self_play(bot: MonteCarloBot, verbose: bool = False):
    """
    Plays a game of self-play using the bot.

    num_moves: The number of moves to cap at
    """

    # Log the moves and policies
    moves = []

    bot.reset_tree()

    # Play the game
    while not bot.mcts.curr.is_terminal() and bot.mcts.curr.move < MAX_MOVES:

        move = bot.choose_move(SEARCHES_PER_MOVE)
        bot.make_move(move)

        moves.append(move)

        if verbose:
            print(f"Move {bot.mcts.curr.move}: {move}")
            print(bot.mcts.curr)

    return (bot.mcts.curr, bot.mcts.curr.compute_winner())


def create_dataset(bot: MonteCarloBot, tqdm_desc: str = "") -> Dataset:
    """
    Creates a new dataset for RL with the specific
    number of games in the config
    """

    out = Dataset()

    for _ in tqdm(range(RL_DS_SIZE), desc=tqdm_desc):
        tree, winner = self_play(bot)
        out.add_rl_game(tree, winner, SELF_PLAY_KEEP_PROB)
    
    return out


def update_dataset(ds: Dataset, bot: MonteCarloBot, tqdm_desc: str = "") -> None:
    """
    Replaces the oldest n sets of game data with
    newly generated game data as specified in the config
    """

    ds.remove_first_n(NEW_GAMES_PER_DS)

    for _ in tqdm(range(NEW_GAMES_PER_DS), desc=tqdm_desc):
        tree, winner = self_play(bot)
        ds.add_rl_game(tree, winner, SELF_PLAY_KEEP_PROB)


def train_one_epoch(ds: Dataset, bot: MonteCarloBot) -> float:
    """
    Trains a bot for one epoch on the produced dataset

    Args:
        ds: Dataset to train on
        bot: bot containing the model to train
    """

    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(bot.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = GoLoss(1) # Balanced cross-entropy and mse

    train_loss = 0
    bot.model = bot.model.to(DEVICE)
    bot.model.train()

    for s, z, pi in dl:
        s = s.to(DEVICE)
        z = z.to(DEVICE)
        pi = pi.to(DEVICE)

        optimizer.zero_grad()

        pi_hat, z_hat = bot.model(s)
        
        loss = criterion(z, z_hat, pi, pi_hat)
        train_loss += loss.item()
        loss.backward()

        optimizer.step()
    
    return train_loss / len(dl)
  

if __name__ == "__main__":
    bot = MonteCarloBot()
    
    print("Building initial dataset... ", end="")
    ds = create_dataset()
    print("Done")

    for i in range(EPOCHS):
        print(f"Epoch {i+1} - Training Loss: ", end="")
        print(train_one_epoch(ds, bot))

        print("Rebuilding dataset... ", end="")
        update_dataset(ds)
        print("Done")

