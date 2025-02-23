import torch.nn as nn
import torch
import game_node as gn

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

class PolicyHead(nn.Module):
    def __init__(self, num_channels: int, board_size=9):
        super().__init__()

        self.conv = nn.Conv2d(num_channels, 2, 1, 1)
        self.norm = nn.BatchNorm2d(2)
        self.relu = nn.ReLU()

        # A fully connected linear layer that outputs a vector of size 9^2 + 1 = 82 corresponding to
        # logit probabilities for all intersections and the pass move
        self.fc = nn.Linear(2*board_size**2, board_size**2 + 1)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
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
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int, padding: int, num_residuals=19):
        super().__init__()

        assert(kernel % 2 == 1) 
        padding = (kernel-1) // 2

        self.conv = ConvBlock(in_channels, out_channels, kernel, stride, padding)
        self.residuals = nn.Sequential(
            *[ResidBlock(out_channels, out_channels, kernel, stride) 
              for _ in range(num_residuals)]
        )

        self.policy_head = PolicyHead(64)
        self.value_head = ValueHead(64)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.residuals(x)

        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value
        
if __name__ == "__main__":
    neural = NeuralNet(2, 3, 1)
    print(neural)

