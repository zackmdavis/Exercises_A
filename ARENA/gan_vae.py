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


# The course implements ConvTranspose2d for us, as support code for the GAN
# that we're building. (I've copy-pasted some support functions/impls rather
# than importing from the instructor's repo.)

def conv2d_minimal(x, weights):
    b, ic, h, w = x.shape
    oc, ic2, kh, kw = weights.shape
    assert ic == ic2, "in_channels for x and weights don't match up"
    ow = w - kw + 1
    oh = h - kh + 1

    s_b, s_ic, s_h, s_w = x.stride()

    x_new_shape = (b, ic, oh, ow, kh, kw)
    x_new_stride = (s_b, s_ic, s_h, s_w, s_h, s_w)

    x_strided = x.as_strided(size=x_new_shape, stride=x_new_stride)

    return einops.einsum(x_strided, weights, "b ic oh ow kh kw, oc ic kh kw -> b oc oh ow")


def force_pair(v: IntOrPair) -> Pair:
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)



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
        stride_h, stride_w = force_pair(self.stride)
        padding_h, padding_w = force_pair(self.padding)

        batch, ic, height, width = x.shape
        ic_2, oc, kernel_height, kernel_width = self.weights.shape
        assert ic == ic_2, f"in_channels for x and weights don't match up. Shapes are {x.shape}, {self.weights.shape}."

        # Apply spacing
        x_spaced_out = fractional_stride_2d(x, stride_h, stride_w)

        # Apply modification (which is controlled by the padding parameter)
        pad_h_actual = kernel_height - 1 - padding_h
        pad_w_actual = kernel_width - 1 - padding_w
        assert min(pad_h_actual, pad_w_actual) >= 0, "total amount padded should be positive"
        x_mod = pad2d(x_spaced_out, left=pad_w_actual, right=pad_w_actual, top=pad_h_actual, bottom=pad_h_actual, pad_value=0)

        # Modify weights
        weights_mod = einops.rearrange(self.weights.flip(-1, -2), "i o h w -> o i h w")

        # Return the convolution
        return conv2d_minimal(x_mod, weights_mod)


    def extra_repr(self) -> str:
        return ", ".join([
            f"{key}={getattr(self, key)}"
            for key in ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
        ])


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim_size: int = 100, # "the size of the latent dimension, i.e. the input to the generator"
        img_size: int = 64, # "the size of the image, i.e. the output of the generator"
        img_channels: int = 3, # "the number of channels in the image (3 for RGB, 1 for grayscale)"
        hidden_channels: List[int] = [128, 256, 512],
            # "the number of channels in the hidden layers of the generator (starting closest
            # to the middle of the DCGAN and going outward, i.e. in chronological order for
            # the generator)"
            #
            # uh, I'm confused that they call `[128, 256, 512]` "chronological
            # order" for the generator, given the diagram that shows the 128 on
            # the side closer to the outputted image ...? But on the other
            # hand, increasing channels from latent to output for the generator
            # makes sense (and GPT-4 confirms this understanding under questioning).
            #
            # Well, at least the "closest to the middle" part makes
            # sense. Maybe it's an erratum.
    ):
        n_layers = len(hidden_channels)
        assert img_size % (2 ** n_layers) == 0, "activation size must double at each layer"

        super().__init__()

        # But what should `project_and_reshape` do? It's supposed to take an
        # 100-element latent vector to a 4×4×1024? GPT-4 suggests that this is
        # going to be a `Linear` ...
        self.project_and_reshape = nn.Linear(latent_dim_size, ((img_size//16) ** 2) * 2 *hidden_channels[-1])
        self.hidden_layers = nn.Sequential(
            # Just filling in the numbers from the diagram, presumably?
            ConvTranspose2d(1024, 512, 8, stride=2),
            ConvTranspose2d(512, 256, 16, stride=2),
            ConvTranspose2d(256, 128, stride=2),
            ConvTranspose2d(128, 64, stride=2),
        )

        return

    def forward(self, x):
        x = self.project_and_reshape(x)
        x = self.hidden_layers(x)
        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        img_channels: int = 3,
        hidden_channels: List[int] = [128, 256, 512],
    ):
        '''
        Implements the discriminator architecture from the DCGAN paper (the mirror image of
        the diagram at the top of page 4). We assume the size of the activations doubles at
        each layer (so image size has to be divisible by 2 ** len(hidden_channels)).

        Args:
            img_size:
                the size of the image, i.e. the input of the discriminator
            img_channels:
                the number of channels in the image (3 for RGB, 1 for grayscale)
            hidden_channels:
                the number of channels in the hidden layers of the discriminator (starting
                closest to the middle of the DCGAN and going outward, i.e. in reverse-
                chronological order for the discriminator)
        '''
        n_layers = len(hidden_channels)
        assert img_size % (2 ** n_layers) == 0, "activation size must double at each layer"

        super().__init__()
        pass

    def forward(self, x: t.Tensor) -> t.Tensor:
        pass


class DCGAN(nn.Module):
    netD: Discriminator
    netG: Generator

    def __init__(
        self,
        latent_dim_size: int = 100,
        img_size: int = 64,
        img_channels: int = 3,
        hidden_channels: List[int] = [128, 256, 512],
    ):
        super().__init__()
        pass
