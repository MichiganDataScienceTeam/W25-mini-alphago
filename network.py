import torch.nn as nn
import torch
from game_node import GameNode
from data_preprocess import node_to_tensor
from config import *

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
    def __init__(self, in_channels: int, board_size=9):
        super().__init__()
        self.board_size = board_size

        # 1. A convolution of 2 filters of kernel size 1 ×1 with stride 1
        # input channels -> 2 channels
        self.conv = nn.Conv2d(in_channels, out_channels=2, kernel_size=1, stride=1)

        # 2. Batch normalisation
        self.bn = nn.BatchNorm2d(2)

        # 3. A rectifier non-linearity
        self.relu = nn.ReLU()

        # 4. A fully connected linear layer, 
        # corresponding to logit probabilities for all intersections and the pass move
        self.fc = nn.Linear(2 * board_size * board_size, board_size * board_size + 1)

        # 5. Convert logits to probabilities
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        # flatten to (batch_size, 2*board_size*board_size)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return self.softmax(x)


class ValueHead(nn.Module):
    def __init__(self, in_channels: int, board_size=9):
        super().__init__()
        self.board_size = board_size

        # 1. A convolution of 1 filter of kernel size 1 ×1 with stride 1
        self.conv = nn.Conv2d(in_channels, out_channels=1, kernel_size=1, stride=1)

        # 2. Batch Normalisation on the single channel output
        self.bn = nn.BatchNorm2d(1)

        # 3. A rectifier non-linearity
        self.relu = nn.ReLU()

        # 4. A fully connected linear layer to a hidden layer of size 256
        self.fc = nn.Linear(board_size * board_size, 256)

        # 5. ReLu again

        # 6. A fully connected linear layer to a scalar
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # 1.
        x = self.conv(x)
        # 2. 
        x = self.bn(x)
        # 3.
        x = self.relu(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # 4.
        x = self.fc(x)
        # 5.
        x = self.relu(x)
        # 6.
        x = self.fc2(x)
        # 7. Tanh non-linearity to ensure output is in [-1, 1]
        return torch.tanh(x)



class NeuralNet(nn.Module):
    def __init__(self, in_channels: int = INPUT_CHANNELS, out_channels: int = OUTPUT_CHANNELS, kernel: int = KERNEL, num_residuals: int = NUM_RESIDUALS):
        super().__init__()

        self.conv = ConvBlock(in_channels, out_channels, kernel)
        
        self.residuals = nn.Sequential(
            *[ResBlock(out_channels, kernel) 
              for _ in range(num_residuals)]
        )

        self.policy_head = PolicyHead(out_channels)
        self.value_head = ValueHead(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.residuals(x)

        policy = self.policy_head(x)
        value = self.value_head(x)

        
        return policy, value

if __name__ == "__main__":
    # Defining board and running game
    board = GameNode(size = 9)

    while not board.is_terminal():
        try:
            print("\nSelect a move")
            row = int(input("Row: "))
            col = int(input("Column: "))

            board = board.create_child((row, col))
        except KeyboardInterrupt:
            print("\nKeyboard Interrupt. Game Ended")
            break
        except:
            print("Error while processing move. Try again.")
        else:
            print(board)
        
    # Make board into tensor
    input_tensor = node_to_tensor(board).unsqueeze(0)

    # Create nn and output
    net_nn = NeuralNet()
    out = net_nn.forward(input_tensor)

    print(out)

    print(out[0].shape)
    print(out[1].shape)

