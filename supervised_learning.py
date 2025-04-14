import torch

from dataset import Dataset
from network import NeuralNet
from config import *
from network import GoLoss

def train(model, epochs=EPOCHS, batch_size=BATCH_SIZE):
    train_ds = Dataset()
    train_ds.load_dir("games")
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_ds = Dataset()
    val_ds.load_dir("val_games")
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = GoLoss(0.01)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}: ", end="")

        train_loss = 0
        val_loss = 0

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

        model.eval()
        for s, z, pi in val_dl:
            s = s.to(DEVICE)
            z = z.to(DEVICE)
            pi = pi.to(DEVICE)

            pi_hat, z_hat = model(s)

            loss = criterion(z, z_hat, pi, pi_hat)
            val_loss += loss.item()

        print(f"Val Loss: {val_loss/len(val_dl)}")
        torch.save(model.state_dict(), f"models/model_epoch_{epoch + 1}.pt")

if __name__ == "__main__":

    # TODO: Make sure you know model interface
    model = NeuralNet().to(DEVICE)
    train(model)
