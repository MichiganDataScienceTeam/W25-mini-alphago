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
        
    def forward(self, x):
        x = self.conv(x)
        x = self.residuals(x)

        return x
        

if __name__ == "__main__":
    current_node = play_game()
    input = traverse_prev(current_node)

    # Converts to float and adds another dimension
    input_tensor = torch.from_numpy(input).float()
    input_tensor = input_tensor.unsqueeze(0)

    resid_nn = ResidBlock(7, 1, 3, 1)
    out = resid_nn.forward(input_tensor)

    print(out)

