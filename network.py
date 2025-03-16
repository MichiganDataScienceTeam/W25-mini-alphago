from typing import Self
import torch.nn as nn
import game_node as GameNode

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
        out = self.norm(out)
        out = self.relu(out)

        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int):
        super().__init__()

        assert(kernel % 2 == 1) 

        padding = (kernel-1) // 2
    
        self.conv_block = ConvBlock(in_channels, out_channels, kernel, stride)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    
    def forward(self, x):
        out = self.conv_block(x)
        out = self.norm1(out)
        out = self.conv(out)
        out = self.norm2(out)
        out += x

        out = self.relu(out)

        return out

    
class PolicyHead(nn.Module):
    def __init__(self, in_channels: int, board_size = 9):
        super().__init__()
    
        self.board_size = board_size
    
        self.conv = nn.Conv2d(in_channels, out_channels=2, kernel=2, stride=1)
        self.bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(2*board_size*board_size, board_size*board_size+1)

    def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            x = x.view(x.size(0), -1) 

            return self.fc(x)
    
class ValueHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int):
        super().__init__()
    
        assert(kernel % 2 == 1) 

        padding = (kernel-1) // 2
    
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding)
        self.fc = nn.Linear()
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(9,1)
        self.fc = nn.Linear()
        self.tanh = nn.Tanh()

    def forward(self, x):
            out = self.conv(x)
            out = self.flatten(out)
            out = self.fc(out)
            out = self.relu(out)
            out = self.flatten(out) 
            out = self.fc(out)
            out = self.tanh(out)


            return out
    
class NeuralNetwork(nn.Module):
    def __init__(self, in_channels: int = 9, out_channels: int=71, kernel: int=3, num_residuals: int = 19):
        super().__init__()

    
    
        self.conv_block = ConvBlock(in_channels, out_channels, kernel)
        self.residual = nn.Sequential(
            *[ResidualBlock(out_channels, kernel) 
              for _ in range(num_residuals)]
                        )
        
        self.policy_head = PolicyHead(out_channels)
        self.value_head = ValueHead(out_channels)
    
    
    def forward(self, x):
        out = self.conv_block(x)
        out = self.residual(x)

        return self.policy_head(out), self.value_head(out)
    
if __name__ == "__main__":
    conv_nn = ConvBlock(9, 1, 3, 1) 

    #TODO: Add code for creating child node of the board, and pass the tensor generated from it through the network


