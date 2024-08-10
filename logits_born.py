import torch
import torch.nn as nn

from bornrule.torch import Born


# check the born_layers.ipynb notebook to see why it's probably a good idea to use this layer as a hidden one, instead of the default Born layer
class BornLogits(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(BornLogits, self).__init__()
        self.born = Born(in_features, out_features, device=device, dtype=dtype)

    def forward(self, x):
        if self.born.is_complex(self.born.weight.dtype):
            return torch.pow(torch.mm(x, self.born.weight).abs(), 2)
        else:
            real = torch.mm(x, self.born.weight[0])
            imag = torch.mm(x, self.born.weight[1])
            return torch.pow(real, 2) + torch.pow(imag, 2)


class LogitsBorn(nn.Module):
    def __init__(self, layer_sizes, device=None, dtype=None):
        super(LogitsBorn, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 2):
            self.layers.append(
                BornLogits(
                    layer_sizes[i], layer_sizes[i + 1], device=device, dtype=dtype
                )
            )

        # final layer uses the original Born to output probabilities
        self.layers.append(
            Born(layer_sizes[-2], layer_sizes[-1], device=device, dtype=dtype)
        )

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = torch.relu(x)
        return self.layers[-1](x)
