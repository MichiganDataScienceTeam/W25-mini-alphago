import os
import platform
import torch
from time import perf_counter
from torch import multiprocessing as mp

from network import NeuralNet, load_model
from bot import MonteCarloBot
from reinforcement_learning import self_play
from dataset import Dataset
from inference_server import InferenceClient, make_buffers, serve

from config import *


TEMP_FILE_PATH = "temp_files"
TEMP_FILE_PREFIX = "temp_ds_file"

# CPUs assigned to this process (SLURM may not assign CPUs 0 to n)
try:
    AVAILABLE_CPUS = sorted(os.sched_getaffinity(0))
except AttributeError:
    AVAILABLE_CPUS = None


def set_start_method_auto() -> None:
    """
    Sets multiprocessing start method
    """

    mp.set_start_method('spawn', force=True)


def worker(model: NeuralNet, games_left: mp.Value, n: int, cpus: set[int], evaluator: InferenceClient = None) -> None:
    """
    Plays self-play games until no games are left and saves
    their positional data to a temp file

    Args:
        model: the neural net (shared memory), unused if evaluator is given
        games_left: shared count of games left to play
        n: the index of this worker
        cpus: the CPUs to run this worker on
        evaluator: client of the inference server (None to use model on cpu)
    """

    torch.set_num_threads(1)
    try:
        os.sched_setaffinity(0, cpus)
    except:
        pass

    # Use cpu for self-play, moving lots of small data is bad
    # Batch evaluations on the inference server instead if available
    bot = MonteCarloBot(model, "cpu", evaluator=evaluator)

    temp_ds = Dataset()

    while True:
        # Take the next game (workers finishing early take more games)
        with games_left.get_lock():
            if games_left.value == 0:
                break

            games_left.value -= 1

        bot.reset()
        tree, winner = self_play(bot)

        temp_ds.add_rl_game(tree, winner)

        print(".", end="", flush=True)

    torch.save(temp_ds, os.path.join(TEMP_FILE_PATH, f"{TEMP_FILE_PREFIX}_{n}.pt"))


def get_cpu(n: int) -> int:
    """
    Returns the nth CPU assigned to this process

    Args:
        n: the index of the CPU
    """

    if AVAILABLE_CPUS is None:
        return n

    return AVAILABLE_CPUS[n % len(AVAILABLE_CPUS)]


def add_games(ds: Dataset, model: NeuralNet, n_processes: int, games_per_process: int, verbose=False) -> None:
    """
    Adds n_processes * games_per_process self-play games
    to the provided Dataset

    Args:
        ds: the Dataset to modify
        model: the NeuralNet to use
        n_processes: the number of processes to use
        games_per_process: the average number of games played by each process
        verbose: whether or not to write logs
    """

    torch.set_num_threads(1)
    try:
        os.sched_setaffinity(0, {get_cpu(0)})
    except:
        pass

    if verbose:
        start_time = perf_counter()

    if not os.path.exists(TEMP_FILE_PATH):
        os.mkdir(TEMP_FILE_PATH)
    elif len(os.listdir(TEMP_FILE_PATH)) > 0:
        raise FileExistsError("Temp file path must start empty if it exists")
    
    print("Multiprocessing Progress:")
    print("X"*(n_processes * games_per_process))

    games_left = mp.Value("i", n_processes * games_per_process)

    # Workers wait on the inference server, so use more workers than CPUs
    if INFERENCE_SERVER:
        n_workers = n_processes * WORKERS_PER_PROCESS

        inputs, outputs = make_buffers(n_workers + 1)
        pipes = [mp.Pipe() for _ in range(n_workers + 1)]

        server = mp.Process(target=serve, args=(model, inputs, outputs, [None] + [pipe[0] for pipe in pipes[1:]]))
        server.start()
    else:
        n_workers = n_processes

    processes = []
    for i in range(1, n_workers+1):
        if INFERENCE_SERVER:
            args = (None, games_left, i, set(AVAILABLE_CPUS or []), InferenceClient(i, inputs, outputs, pipes[i][1]))
        else:
            args = (model, games_left, i, {get_cpu(i)})

        p = mp.Process(target=worker, args=args)
        p.start()
        processes.append(p)

    # Close copies of the pipes so the server stops once all workers finish
    if INFERENCE_SERVER:
        for server_end, worker_end in pipes:
            server_end.close()
            worker_end.close()
    
    for p in processes:
        p.join()

    # Fail loudly instead of silently losing games
    failed = [i for i, p in enumerate(processes, 1) if p.exitcode != 0]

    if INFERENCE_SERVER:
        server.join()

        if server.exitcode != 0 and not failed:
            raise RuntimeError("Inference server failed (see traceback above)")

    if failed:
        raise RuntimeError(f"Self-play workers {failed} failed (see traceback above)")

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
    create_dataset("RL_weights.pt", 4, 1, True)
