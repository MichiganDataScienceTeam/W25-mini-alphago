import torch.nn as nn
import torch
from preprocessing import *

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int):
        super().__init__()

        assert(kernel % 2 == 1) 

        padding = (kernel-1) // 2
    
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    
    def forward(self, x):
        out = self.conv(x)
        print(out.shape)
        out = self.norm(out)
        out = self.relu(out)

        return out
    
class ResidBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int):
        super().__init__()

        assert(kernel % 2 == 1) 

        padding = (kernel-1) // 2
        self.conv_block = ConvBlock(in_channels, out_channels, kernel, stride)
        
        self.conv = nn.Conv2d(out_channels, out_channels, kernel, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Create instance of conv_block and run through, this is 'f(x)'
        g_x = self.conv_block(x)
        g_x = self.conv(g_x)
        g_x = self.norm(g_x)
        g_x = self.relu(x + g_x)
        
        return g_x
        
class NeuralNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int, num_residuals=19):
        super().__init__()

        assert(kernel % 2 == 1) 

        self.conv = ConvBlock(in_channels, out_channels, kernel, stride)
        
        self.residuals = nn.Sequential(
            *[ResidBlock(out_channels, out_channels, kernel, stride) 
              for _ in range(num_residuals)]
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = self.residuals(x)

        return x
    
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
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        # flatten to (batch_size, 2*board_size*board_size)
        x = x.view(x.size(0), -1)
        
        return self.fc(x)
    
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

        

if __name__ == "__main__":
    current_node = play_game()
    input = traverse_prev(current_node)

    # Converts to float and adds another dimension
    input_tensor = torch.from_numpy(input).float()
    input_tensor = input_tensor.unsqueeze(0)

    net_nn = NeuralNet(7, 71, 3, 1)
    out = net_nn.forward(input_tensor)

    policy = PolicyHead(71)
    out_policy = policy.forward(out)

    value = ValueHead(71)
    out_value = value.forward(out)

    print(out_policy)
    print(out_value)

