import os
import platform
import torch
from time import perf_counter
from torch import multiprocessing as mp

from network import NeuralNet, load_model
from bot import MonteCarloBot
from reinforcement_learning import self_play
from dataset import Dataset

from config import *


TEMP_FILE_PATH = "temp_files"
TEMP_FILE_PREFIX = "temp_ds_file"


def set_start_method_auto() -> None:
    """
    Sets multiprocessing start method
    """

    mp.set_start_method('spawn', force=True)


def worker(model: NeuralNet, games: int, n: int) -> float:
    """
    Runs one self-play game and saves its positional data
    to the specified filepath and returns the time elapsed

    Args:
        model: the neural net (shared memory)
        n: the index of this worker
    """

    torch.set_num_threads(1)
    os.sched_setaffinity(0, {n})

    # Use cpu for self-play, moving lots of small data is bad
    # Also, the multiprocessing becomes very demanding if shared gpu
    bot = MonteCarloBot(model, "cpu")

    temp_ds = Dataset()

    for _ in range(games):
        bot.reset_tree()
        tree, winner = self_play(bot)

        temp_ds.add_rl_game(tree, winner)

        print(n, end="", flush=True)

    torch.save(temp_ds, os.path.join(TEMP_FILE_PATH, f"{TEMP_FILE_PREFIX}_{n}.pt"))


def add_games(ds: Dataset, model: NeuralNet, n_processes: int, games_per_process: int, verbose=False) -> None:
    """
    Adds n_processes * games_per_process self-play games
    to the provided Dataset

    Args:
        ds: the Dataset to modify
        model: the NeuralNet to use
        n_processes: the number of processes to use
        games_per_process: the number of games played by each process
        verbose: whether or not to write logs
    """

    torch.set_num_threads(1)
    os.sched_setaffinity(0, {0})

    if verbose:
        start_time = perf_counter()

    if not os.path.exists(TEMP_FILE_PATH):
        os.mkdir(TEMP_FILE_PATH)
    elif len(os.listdir(TEMP_FILE_PATH)) > 0:
        raise FileExistsError("Temp file path must start empty if it exists")
    
    print("Multiprocessing Progress:")
    print("X"*(n_processes * games_per_process))

    processes = []
    for i in range(1, n_processes+1):
        p = mp.Process(target=worker, args=(model, games_per_process, i))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

    for filename in os.listdir(TEMP_FILE_PATH):
        if not filename.startswith(TEMP_FILE_PREFIX):
            raise FileExistsError(f"Temp file directory must only have temp files. Found {filename} instead.")

        filepath = os.path.join(TEMP_FILE_PATH, filename)

        ds.merge(torch.load(filepath, weights_only=False))

        os.remove(filepath)
    
    if verbose:
        print(f"\n  Finished in {perf_counter() - start_time}s")


def create_dataset(model_path: str, n_processes: int, games_per_process: int, verbose: bool = False) -> Dataset:
    """
    Uses multiprocessing to generate a self-play dataset

    Args:
        model_path: a path to a saved NeuralNet
        n_processes: the number of processes to use
        games_per_process: the number of games played by each process
        verbose: whether or not to write logs
    """

    set_start_method_auto()

    ds = Dataset()

    model = load_model(model_path, "  Multiprocessing: Model loaded from")
    model.eval()
    model.share_memory()

    add_games(ds, model, n_processes, games_per_process, verbose)

    return ds


def update_dataset(ds: Dataset, model_path: str, n_processes: int, games_per_process: int, verbose: bool = False):
    """
    Uses multiprocessing to update a self-play dataset

    Args:
        ds: the Dataset to update
        model_path: a path to a saved NeuralNet
        n_processes: the number of processes to use
        games_per_process: the number of games played by each process
        verbose: whether or not to write logs
    """

    ds.remove_first_n(n_processes * games_per_process)

    set_start_method_auto()

    model = load_model(model_path, "  Multiprocessing: Model loaded from")
    model.eval()
    model.share_memory()

    add_games(ds, model, n_processes, games_per_process, verbose)


if __name__ == "__main__":
    create_dataset("Great_Lakes_Weights.pt", 4, 1, True)
