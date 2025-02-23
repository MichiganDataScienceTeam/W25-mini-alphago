import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int):
        super().__init__()
    
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            stride=1,
            padding="same"
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)

        return out
    

class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel: int):
        super().__init__()

        conv_kwargs = {
            "in_channels": channels,
            "out_channels": channels,
            "kernel_size": kernel,
            "stride": 1,
            "padding": "same"
        }

        self.conv1 = nn.Conv2d(**conv_kwargs)
        self.norm1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(**conv_kwargs)
        self.norm2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU()
    
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        
        out += x

        out = self.relu(out)

        return out


class PolicyHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(9*9+1, 9*9+1)
        self.softmax = nn.Softmax(dim=1)
    

    def forward(self, x):
        # Return tensor in the shape of (batch, 9*9 + 1)
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.fc(out)
        out = self.softmax(out)

        return out


class ValueHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(9*9, 1)
        self.tanh = nn.Tanh()
    

    def forward(self, x):
        # Return tensor in the shape of (batch, 1)
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.tanh(out)

        return out


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize members

        raise NotImplementedError()
    

    def forward(self, x):
        # Return tuple of tensors in the shape ((batch, 9*9 + 1), (batch, 1))

        raise NotImplementedError()


if __name__ == "__main__":
    model = NeuralNet()

    #TODO: Add code for creating child node of the board, and pass the tensor generated from it through the network

