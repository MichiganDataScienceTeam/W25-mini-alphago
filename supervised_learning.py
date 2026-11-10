import os
import random
import torch

from typing import Tuple

from dataset import Dataset
from network import NeuralNet
from config import *
from network import GoLoss, save_model


def load_datasets(val_fraction: float = SL_VAL_FRACTION, augment: bool = SL_AUGMENT, seed: int = 0) -> Tuple[Dataset, Dataset]:
    """
    Returns the training and validation Datasets

    The validation set is all of val_games/ plus a fixed random
    fraction of games/, which is never trained on

    Args:
        val_fraction: fraction of games/ to hold out for validation
        augment: whether to add all 8 symmetries of each training game
        seed: seed for the random hold out
    """

    filepaths = [os.path.join("games", game_file) for game_file in sorted(os.listdir("games"))]
    random.Random(seed).shuffle(filepaths)

    n_val = int(len(filepaths) * val_fraction)

    train_ds = Dataset()
    train_ds.load_files(filepaths[n_val:], augment)

    val_ds = Dataset()
    val_ds.load_files(filepaths[:n_val])
    val_ds.load_dir("val_games")

    return train_ds, val_ds


def evaluate(model: NeuralNet, dl: torch.utils.data.DataLoader, criterion: GoLoss) -> Tuple[float, float]:
    """
    Returns the mean loss and policy accuracy (how often the
    most likely move is the human move) of the model

    Args:
        model: the NeuralNet to evaluate
        dl: DataLoader of the data to evaluate on
        criterion: the loss function
    """

    total_loss = 0
    n_correct = 0

    model.eval()
    with torch.no_grad():
        for s, z, pi in dl:
            s = s.to(DEVICE)
            z = z.to(DEVICE)
            pi = pi.to(DEVICE)

            pi_hat, z_hat = model(s)

            loss = criterion(z, z_hat, pi, pi_hat)
            total_loss += loss.item()

            n_correct += (pi_hat.argmax(dim=1) == pi.argmax(dim=1)).sum().item()

    return total_loss / len(dl), n_correct / len(dl.dataset)


def train(model, epochs=SL_EPOCHS, batch_size=BATCH_SIZE, save_path="SL_weights.pt"):
    """
    Trains the model on human games and saves
    the model with the lowest validation loss

    Args:
        model: the NeuralNet to train (on DEVICE)
        epochs: the number of epochs to train for
        batch_size: the batch size
        save_path: the file to save the best model to
    """

    train_ds, val_ds = load_datasets()
    print(f"Train positions: {len(train_ds)}, Val positions: {len(val_ds)}")

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = GoLoss(0.01)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}: ", end="")

        train_loss = 0

        model.train()
        for s, z, pi in train_dl:
            s = s.to(DEVICE)
            z = z.to(DEVICE)
            pi = pi.to(DEVICE)

            optimizer.zero_grad()

            pi_hat, z_hat = model(s)

            loss = criterion(z, z_hat, pi, pi_hat)
            train_loss += loss.item()
            loss.backward()

            optimizer.step()

        print(f"Train Loss: {train_loss/len(train_dl)} ", end="")

        val_loss, val_acc = evaluate(model, val_dl, criterion)

        print(f"Val Loss: {val_loss} Val Acc: {val_acc}")

        # Keep the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, save_path, "  Best model saved to")


if __name__ == "__main__":
    model = NeuralNet().to(DEVICE)
    train(model)
