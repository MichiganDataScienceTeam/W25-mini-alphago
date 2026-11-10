import numpy as np
import torch

from multiprocessing.connection import Connection, wait
from numpy.typing import NDArray
from typing import List, Tuple

from network import NeuralNet

from config import *


class InferenceClient:
    """
    Evaluates positions by sending them to the inference server
    (used as the MonteCarlo evaluator in self-play workers)

    Positions and results are passed through shared memory, so the
    pipe only carries a tiny signal per request

    Args:
        n: the index of this client (its slot in inputs and outputs)
        inputs: shared tensor of positions, one per client
        outputs: shared tensor of [value, *policy], one per client
        conn: this client's end of its pipe to the server
    """


    def __init__(self, n: int, inputs: torch.Tensor, outputs: torch.Tensor, conn: Connection):
        self.n = n
        self.inputs = inputs
        self.outputs = outputs
        self.conn = conn


    def __call__(self, x: NDArray) -> Tuple[float, NDArray]:
        """
        Returns the network's (value, policy) for the position

        Args:
            x: the position from node_to_tensor as a numpy array
        """

        self.inputs[self.n] = torch.from_numpy(x)

        self.conn.send_bytes(b"")
        self.conn.recv_bytes()

        out = self.outputs[self.n]

        return out[0].item(), out[1:].numpy().copy()


def make_buffers(n_clients: int, size: int = 9) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the shared (inputs, outputs) tensors for n_clients clients

    Args:
        n_clients: the number of clients (indexed from 0)
        size: the board size
    """

    inputs = torch.zeros((n_clients, INPUT_CHANNELS, size, size)).share_memory_()
    outputs = torch.zeros((n_clients, size * size + 2)).share_memory_()

    return inputs, outputs


def serve(model: NeuralNet, inputs: torch.Tensor, outputs: torch.Tensor, conns: List[Connection], device: str = DEVICE) -> None:
    """
    Evaluates positions from all clients in batches until
    every client has closed its pipe

    Args:
        model: the neural net (shared memory)
        inputs: shared tensor of positions, one per client
        outputs: shared tensor of [value, *policy], one per client
        conns: server end of each client's pipe (indexed by client)
        device: the device to evaluate on
    """

    torch.set_num_threads(1)

    model = model.to(device)
    model.eval()

    # Client index of each open pipe
    open_conns = {conn: n for n, conn in enumerate(conns) if conn is not None}

    while len(open_conns) > 0:
        ns = []

        # Batch every client that is waiting
        for conn in wait(list(open_conns)):
            try:
                conn.recv_bytes()
                ns.append(open_conns[conn])
            except EOFError:
                # Client finished
                del open_conns[conn]

        if len(ns) == 0:
            continue

        with torch.inference_mode():
            policy, value = model(inputs[ns].to(device))

        outputs[ns, 0] = value[:, 0].float().cpu()
        outputs[ns, 1:] = policy.float().cpu()

        for n in ns:
            conns[n].send_bytes(b"")
