import torch
from torch import nn

# We start with some simple module implementations ...

class Tanh(nn.Module):
    def forward(self, x):
        return torch.tanh(x)


class LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope
        # I initially forgot the `super().__init__()`, and was confused why I
        # had to add some lines like `self._backward_hooks = []` for missing
        # attributes in order to get the test to pass.

    def forward(self, x):
        # I wasn't sure what was wrong with my solution, and it turned out that
        # I had forgotten the `return` keyword—perils of writing Rust last
        # week?
        return torch.maximum(x, torch.tensor(0.0)) + self.negative_slope * torch.minimum(x, torch.tensor(0.0))
        # Instructor's solution was—
        # return t.where(x > 0, x, self.negative_slope * x)
        # which instructs us on `torch.where` (an if–then–else construct)

    def extra_repr(self):
        return "negative_slope={}".format(self.negative_slope)


class Sigmoid(nn.Module):
    def forward(self):
        return torch.sigmoid(x)


# The course implements this for us, as support code for the GAN that we're building.
class ConvTranspose2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0
    ):
        '''
        Same as torch.nn.ConvTranspose2d with bias=False.
        '''
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        kernel_size = force_pair(kernel_size)
        sf = 1 / (self.out_channels * kernel_size[0] * kernel_size[1]) ** 0.5

        self.weight = nn.Parameter(sf * (2 * torch.rand(in_channels, out_channels, *kernel_size) - 1))

    def forward(self, x):
        return solutions.conv_transpose2d(x, self.weight, self.stride, self.padding)

    def extra_repr(self) -> str:
        return ", ".join([
            f"{key}={getattr(self, key)}"
            for key in ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
        ])
