import torch
import torch.nn as nn

class ResBlock(nn.Sequential):
    def __init__(self, num_channels, kernel_size, norm_layer):
        super(ResBlock, self).__init__()
        layers = []
        layers += [
            nn.ReflectionPad2d((kernel_size - 1) // 2),
            nn.Conv2d(num_channels, num_channels, kernel_size, bias=False),
            norm_layer(num_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d((kernel_size - 1) // 2),
            nn.Conv2d(num_channels, num_channels, kernel_size, bias=False),
            norm_layer(num_channels, affine=True),
        ]        
        self.blocks = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.blocks(x) + x


class ResNet(nn.Module):
    def __init__(self, input_channels, output_channels, num_blocks, num_channels, kernel_size,  norm_layer=nn.BatchNorm2d):
        super(ResNet, self).__init__()

        layers = []
        layers += [
            nn.ReflectionPad2d((kernel_size - 1) // 2),
            nn.Conv2d(input_channels, num_channels, kernel_size),
            nn.LeakyReLU(0.2, inplace=True)]

        for i in range(num_blocks):
            layers += [ResBlock(num_channels, kernel_size, norm_layer)]
       
        layers += [
            nn.ReflectionPad2d((kernel_size - 1) // 2),
            nn.Conv2d(num_channels, num_channels, kernel_size),
            norm_layer(num_channels, affine=True)]

        layers += [
            nn.ReflectionPad2d((kernel_size - 1) // 2),
            nn.Conv2d(num_channels, output_channels, kernel_size),
            nn.Sigmoid()]
            
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)
