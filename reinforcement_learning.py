import numpy as np
import torch

from tqdm import tqdm
from typing import List, Tuple

from bot import MonteCarloBot
from dataset import Dataset
from network import GoLoss
from tree_node import TreeNode

from config import *


def self_play(bot: MonteCarloBot, verbose: bool = False):
    """
    Plays a game of self-play using the bot.

    Args:
        verbose: toggle display every move
    """

    # Log the moves and policies
    moves = []

    bot.reset()

    # Play the game
    while not bot.mcts.curr.is_terminal() and bot.mcts.curr.move < MAX_MOVES:

        # Explore moves the network doesn't like yet
        bot.mcts.add_noise()

        new_curr = bot.choose_move_tree(SEARCHES_PER_MOVE)
        bot.mcts.curr = new_curr
        move = new_curr.prev_move

        moves.append(move)

        if verbose:
            print(f"Move {bot.mcts.curr.move}: {move}")
            print(bot.mcts.curr)

    return (bot.mcts.curr, bot.mcts.curr.compute_winner())


def game_stats(node: TreeNode, winner: int) -> Tuple[int, int, int]:
    """
    Returns (number of moves, winner, move number of the first pass or 0)
    of a self-play game

    Args:
        node: the last TreeNode in the game
        winner: the winner from compute_winner
    """

    first_pass = 0
    curr = node

    while curr.prev is not None:
        if curr.prev_move == (-1, -1):
            first_pass = curr.move

        curr = curr.prev

    return node.move, winner, first_pass


def print_stats(stats: List[Tuple[int, int, int]]) -> None:
    """
    Prints a summary of self-play games to catch problems early
    (like passing as soon as allowed or games never ending)

    Args:
        stats: list of game_stats for each game
    """

    moves, winners, first_passes = (np.array(x) for x in zip(*stats))
    passed = first_passes[first_passes > 0]

    print(f"  Self-play stats ({len(stats)} games):")
    print(f"    Average length: {moves.mean():.1f} moves (min {moves.min()}, max {moves.max()})")
    print(f"    Ended as soon as passing was allowed: {np.mean(moves <= NUM_MOVES_ALLOW_PASS + 2):.0%}")
    print(f"    Hit MAX_MOVES: {np.mean(moves >= MAX_MOVES):.0%}")
    print(f"    Black win rate: {np.mean(winners == 1):.0%}")
    print(f"    Average first pass: move {passed.mean() if len(passed) > 0 else 0:.1f}")


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


def make_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Creates the optimizer for RL training

    Reuse the same optimizer across epochs to keep its state

    Args:
        model: the NeuralNet to optimize
    """

    return torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)


def train_one_epoch(ds: Dataset, bot: MonteCarloBot, optimizer: torch.optim.Optimizer = None) -> float:
    """
    Trains a bot for one epoch on the produced dataset

    Args:
        ds: Dataset to train on
        bot: bot containing the model to train
        optimizer: optimizer from make_optimizer (creates a new one if None)
    """

    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    if optimizer is None:
        optimizer = make_optimizer(bot.model)

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
    ds = create_dataset(bot)
    print("Done")

    optimizer = make_optimizer(bot.model)

    for i in range(EPOCHS):
        print(f"Epoch {i+1} - Training Loss: ", end="")
        print(train_one_epoch(ds, bot, optimizer))

        print("Rebuilding dataset... ", end="")
        update_dataset(ds, bot)
        print("Done")

