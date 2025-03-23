import torch
from dataset import Dataset
from network import NeuralNetwork
from config import *

class GoLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, z_hat, pi, pi_hat):
        """ z is the value of the game, pi is the policy vector, z_hat and pi_hat are the predicted values """
        value_loss = torch.nn.functional.mse_loss(z, z_hat)
        policy_loss = torch.nn.functional.cross_entropy(pi_hat, pi)

        return value_loss + policy_loss


def train(model, epochs=EPOCHS, batch_size=BATCH_SIZE):
    dataset = Dataset("games")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = GoLoss()

    for epoch in range(epochs):
        for i, (s, z, pi) in enumerate(dataloader):
            optimizer.zero_grad()

            pi_hat, z_hat = model(s)

            loss = criterion(z.to(torch.float32), z_hat, pi, pi_hat)
            loss.backward()

            optimizer.step()

            print(f"Epoch {epoch}, Batch {i}: Loss: {loss.item()}")




if __name__ == "__main__":

    # TODO: Make sure you know model interface
    model = NeuralNet(7, 7, 3).float()
    train(model)
