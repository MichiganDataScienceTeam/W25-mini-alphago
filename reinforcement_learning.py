import numpy as np
import random
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

        # Only full searches are good enough to train the policy on
        full_search = random.random() < FULL_SEARCH_PROB

        if full_search:
            # Explore moves the network doesn't like yet
            bot.mcts.add_noise()
            bot.mcts.curr.full_search = True

        new_curr = bot.choose_move_tree(FULL_SEARCHES if full_search else FAST_SEARCHES)
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

    for _ in tqdm(range(RL_DS_START_SIZE), desc=tqdm_desc):
        tree, winner = self_play(bot)
        out.add_rl_game(tree, winner)
    
    return out


def update_dataset(ds: Dataset, bot: MonteCarloBot, tqdm_desc: str = "") -> None:
    """
    Adds newly generated game data as specified in the config,
    removing the oldest games once the dataset is full
    """

    n_remove = len(ds.start_indices) + NEW_GAMES_PER_DS - RL_DS_SIZE

    if n_remove > 0:
        ds.remove_first_n(n_remove)

    for _ in tqdm(range(NEW_GAMES_PER_DS), desc=tqdm_desc):
        tree, winner = self_play(bot)
        ds.add_rl_game(tree, winner)


def make_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Creates the optimizer for RL training

    Reuse the same optimizer across epochs to keep its state

    Args:
        model: the NeuralNet to optimize
    """

    return torch.optim.AdamW(model.parameters(), lr=RL_LEARNING_RATE, weight_decay=1e-4)


def random_symmetry(s: torch.Tensor, pi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the batch with each position and its policy rotated and/or
    reflected by a random one of the 8 symmetries of the board

    Args:
        s: batch of positions from node_to_tensor
        pi: batch of policies (pass last)
    """

    size = s.shape[-1]

    s = s.clone()
    board_pi = pi[:, :-1].reshape(-1, size, size).clone()

    symmetries = torch.randint(0, 8, (len(s),))

    for t in range(1, 8):
        idx = symmetries == t

        def transform(x):
            x = torch.rot90(x, t % 4, dims=(-2, -1))
            return x.flip(-1) if t >= 4 else x

        s[idx] = transform(s[idx])
        board_pi[idx] = transform(board_pi[idx])

    return s, torch.cat([board_pi.reshape(len(pi), -1), pi[:, -1:]], dim=1)


def evaluate_loss(positions: list, model: torch.nn.Module) -> Tuple[float, float, float]:
    """
    Returns the mean (total, value, policy) loss on positions without training

    Use on new games before training on them, a loss much higher
    than the training loss means the model is memorizing old games

    Args:
        positions: list of (s, z, pi) from a Dataset
        model: the NeuralNet to evaluate
    """

    dl = torch.utils.data.DataLoader(positions, batch_size=BATCH_SIZE)

    criterion = GoLoss(1)

    total_loss = value_loss = policy_loss = 0
    model = model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        for s, z, pi in dl:
            pi_hat, z_hat = model(s.to(DEVICE))

            total_loss += criterion(z.to(DEVICE), z_hat, pi.to(DEVICE), pi_hat).item()
            value_loss += criterion.value_loss.item()
            policy_loss += criterion.policy_loss.item()

    return total_loss / len(dl), value_loss / len(dl), policy_loss / len(dl)


def train_one_epoch(ds: Dataset, bot: MonteCarloBot, optimizer: torch.optim.Optimizer = None) -> Tuple[float, float, float]:
    """
    Trains a bot for one epoch on the produced dataset and
    returns the mean (total, value, policy) loss

    Args:
        ds: Dataset to train on
        bot: bot containing the model to train
        optimizer: optimizer from make_optimizer (creates a new one if None)
    """

    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    if optimizer is None:
        optimizer = make_optimizer(bot.model)

    criterion = GoLoss(1) # Balanced cross-entropy and mse

    train_loss = value_loss = policy_loss = 0
    bot.model = bot.model.to(DEVICE)
    bot.model.train()

    for s, z, pi in dl:
        if RL_AUGMENT:
            s, pi = random_symmetry(s, pi)

        s = s.to(DEVICE)
        z = z.to(DEVICE)
        pi = pi.to(DEVICE)

        optimizer.zero_grad()

        pi_hat, z_hat = bot.model(s)
        
        loss = criterion(z, z_hat, pi, pi_hat)
        train_loss += loss.item()
        value_loss += criterion.value_loss.item()
        policy_loss += criterion.policy_loss.item()
        loss.backward()

        optimizer.step()
    
    return train_loss / len(dl), value_loss / len(dl), policy_loss / len(dl)
  

if __name__ == "__main__":
    bot = MonteCarloBot()
    
    print("Building initial dataset... ", end="")
    ds = create_dataset(bot)
    print("Done")

    optimizer = make_optimizer(bot.model)

    for i in range(EPOCHS):
        print(f"Epoch {i+1} - Training Loss (total, value, policy): ", end="")
        print(train_one_epoch(ds, bot, optimizer))

        print("Rebuilding dataset... ", end="")
        update_dataset(ds, bot)
        print("Done")

