import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, num_in_channels: int, num_out_channels: int, kernel: int, stride: int):
        super().__init__()

        assert(kernel % 2 == 1) 

        padding = (kernel-1) // 2
    
        self.conv = nn.Conv2d(num_in_channels, num_out_channels, kernel, stride, padding)
        self.norm = nn.BatchNorm2d(num_out_channels)
        self.relu = nn.ReLU()


    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)

        return out


class ResBlock(nn.Module):
    def __init__(self, num_channels: int, kernel: int, stride: int):
        super().__init__()

        padding = (kernel-1) // 2

        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel, stride, padding)
        self.norm1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel, stride, padding)
        self.norm2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU()


    def forward(self, x):
        """ based on: https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf#page=27 """
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + x
        out = self.relu(out)

        return out


class PolicyHead(nn.Module):
    def __init__(self, num_channels: int, board_size=9):
        super().__init__()

        self.conv = nn.Conv2d(num_channels, 1, 3, 1, 'same')
        self.norm = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()

        # A fully connected linear layer that outputs a vector of size 9^2 + 1 = 82 corresponding to
        # logit probabilities for all intersections and the pass move
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(board_size**2, board_size**2 + 1)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.flatten(out)
        out = self.fc(out)

        return out


class ValueHead(nn.Module):
    def __init__(self, num_channels: int, board_size=9):
        super().__init__()

        self.conv = nn.Conv2d(num_channels, 1, 1, 1)
        self.norm = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(board_size**2, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.tanh(out)

        return out


class NeuralNet(nn.Module):
    def __init__(self, lookback: int, kernel: int, stride: int):
        super().__init__()

        self.convolution = ConvBlock(lookback*2+1, 64, kernel, stride)
        self.res_tower = nn.Sequential(*[ResBlock(64, kernel, stride) for _ in range(20)])  # 39 layers in the original
        self.policy_head = PolicyHead(64)
        self.value_head = ValueHead(64)


    def forward(self, x):
        """ based on: https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf#page=27 """
        out = self.convolution(x)
        for res in self.res_tower:
            out = res(out)
        policy = self.policy_head(out)
        value = self.value_head(out)

        return policy, value


if __name__ == "__main__":
    # conv_nn = ConvBlock(9, 1, 3, 1)
    neural = NeuralNet(2, 3, 1)
    print(neural)

    #TODO: Add code for creating child node of the board, and pass the tensor generated from it through the network

