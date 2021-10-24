import torch
import torch.nn as nn

class MultiScale(nn.Module):
    def __init__(self, scales, in_channels, out_channels):
        super(MultiScale, self).__init__()
        self._scales = scales
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._pooling = nn.ModuleList()
        for i in range(self._scales-1):
            self._pooling.append(nn.Conv2d(in_channels=self._in_channels, out_channels=self._out_channels, kernel_size=(1, 3), padding=(0, 1),
                         stride=(1, 2)))
    

    def forward(self, x):
        scale = []
        scale.append(x)
        for i in range(self._scales-1):
            x = self._pooling[i](x)
            scale.append(x)
        
        return scale
