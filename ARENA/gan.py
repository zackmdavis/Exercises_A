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


def force_pair(v):
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



class Discriminator(nn.Module):
    def __init__(
        self,
        img_size=64,
        img_channels=3,
        hidden_channels=[128, 256, 512, 1024],
    ):
        n_layers = len(hidden_channels)
        assert img_size % (2 ** n_layers) == 0, "activation size must double at each layer"

        super().__init__()

        channel_sequence = [img_channels] + hidden_channels

        convolutional_layers = []
        for i in range(len(channel_sequence) - 1):
            convolutional_layers.append(
                nn.Conv2d(channel_sequence[i], channel_sequence[i+1], kernel_size=4, stride=2, padding=1, bias=False)
            )
            if i != 0: # "All blocks have a batchnorm layer, except for the very first one."
                convolutional_layers.append(nn.BatchNorm2d(channel_sequence[i+1]))
            convolutional_layers.append(LeakyReLU(0.2))

        self.convolve = nn.Sequential(*convolutional_layers)

        self.predict = nn.Sequential(*[
            # So at the end of the convolutional blocks, we have a batch×1024×4×4, which we flatten to batch×16384
            nn.Flatten(),
            # and then use a fully-connected layer to map it to batch×1
            nn.Linear(channel_sequence[-1] * (img_size//(2 ** n_layers))**2, 1, bias=False),
            # and then sigmoid it to a probability.
            Sigmoid(),
        ])

    def forward(self, x):
        return self.predict(self.convolve(x))


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim_size=100,
        img_size=64,
        img_channels=3,
        hidden_channels=[128, 256, 512, 1024],
    ):
        n_layers = len(hidden_channels)
        assert img_size % (2 ** n_layers) == 0, "activation size must double at each layer"

        super().__init__()

        channel_sequence = list(reversed(hidden_channels)) + [img_channels]

        self.project_and_reshape = nn.Sequential(
            # fully-connected layer from batch×latent_dim_size to batch×16384
            nn.Linear(latent_dim_size, channel_sequence[0] * (img_size//(2 ** n_layers))**2, bias=False),
            # shaped as 1024×4×4
            nn.Unflatten(1, (channel_sequence[0], img_size//(2 ** n_layers), img_size//(2 ** n_layers))),
            nn.BatchNorm2d(channel_sequence[0]),
            nn.ReLU(),
        )

        transconvolutional_layers = []
        for i in range(len(channel_sequence) - 1):
            transconvolutional_layers.append(
                ConvTranspose2d(channel_sequence[i], channel_sequence[i+1], kernel_size=4, stride=2, padding=1),
            )
            # last layer has no BatchNorm and tanh instead of ReLU
            if i != len(channel_sequence) - 2:
                transconvolutional_layers.extend([
                    nn.BatchNorm2d(channel_sequence[i+1]),
                    nn.ReLU(),
                ])
            else:
                transconvolutional_layers.extend([
                    Tanh()
                ])

        self.generate = nn.Sequential(*transconvolutional_layers)

    def forward(self, x):
        return self.generate(self.project_and_reshape(x))


# OK, after finally reading enough to understand what's going on, it's time to
# compare to the solutions and then train it.
#
# I didn't end up with the same parameter counts as the canonical solutions
# (and I have a different `hidden_channels` default, because the diagram and
# the explanatory text seemed to contradict each other, and I went with the
# diagram), but at least the sequence of layers looks the same?

# My gan_vae.Discriminator(hidden_channels=[128, 256, 512]) has 2,759,041
# params; instructor's has 2,661,888.

# The Conv2D/BatchNorm in the middle match, the next Conv2D is almost the
# same—difference of 512 probably a matter of me using the torch.nn version and
# needing to pass bias=False?. But in my first Conv2d, I have 131,200
# vs. instructor's 6,144, and in my linear at the end, I have 4,097
# vs. instructor's 32,768.

# So what happened? I now insert the `bias=False` kwargs, and it turns out that
# I forgot to square the final downsampled sized in the Linear at the end.
# Then the only remaining mismatch is, why is my first Conv2d layer so big
# (131,072 vs. expected 6,144)? Can `torchinfo` give me more details than the param count?

# GPT-4 suggests `model.modules()` instead of conjectured `torchinfo`
# functionality.  I had a `Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2),
# padding=(1, 1), bias=False)` where the instructor's soultion had a
# `Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2,
# padding=1)`—I see, because I used `img_size` instead of `img_channels` at the
# start of my channel sequence.

# Embarrassingly late lightbulb moment: a convolutional layer doesn't need to
# "know" about big the image is, does it? Because it's only learning weights
# for the kernel which gets slid over the image?

# OK, now the discriminators match! What about the generators? The only
# difference seems to be that I have many fewer parameters in my first linear
# layer. Mine is a `Linear(in_features=100, out_features=4096, bias=False)`;
# instructor's is a `Linear(in_features=100, out_features=32768,
# bias=False)`—because I forgot to square again.


# Now it's time to train. We get a zipfile with a lot of celebrity faces, and
# code to load either that or MNIST digits.

# The face images are taking a long time to unzip! It took about twenty minutes
# to get halfway!
#
# zmd@system76-pc:~/Code/Exercises_A/ARENA$ unzip -l img_align_celeba.zip  | wc -l
# 202605
# zmd@system76-pc:~/Code/Exercises_A/ARENA$ ll data/celeba/img_align_celeba/ -1 | wc -l
# 104406

import pathlib

from torchvision import datasets, transforms

def get_dataset(dataset, train):
    assert dataset in ["MNIST", "CELEB"]

    if dataset == "CELEB":
        image_size = 64
        assert train, "CelebA dataset only has a training set"
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        trainset = datasets.ImageFolder(
            root = pathlib.Path.cwd() / "data" / "celeba",
            transform = transform
        )

    elif dataset == "MNIST":
        img_size = 28
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = datasets.MNIST(
            root = pathlib.Path.cwd() / "data",
            transform = transform,
            download = True,
        )

    return trainset


# Well, there was also a weight-initialization exercise. Let's do that on our DCGAN wrapper class.

class DCGAN(nn.Module):
    def __init__(
        self,
        latent_dim_size=100,
        img_size=64,
        img_channels=3,
        hidden_channels=[128, 256, 512],
    ):
        super().__init__()
        self.latent_dim_size = latent_dim_size
        self.img_size = img_size
        self.img_channels = img_channels
        self.hidden_channels = hidden_channels
        self.discriminator = Discriminator(img_size, img_channels, hidden_channels)
        self.generator = Generator(latent_dim_size, img_size, img_channels, hidden_channels)

    def initialize_weights(self):
        for network in [self.discriminator, self.generator]:
            for module in network.modules():
                # This is gross (maybe I should have been more careful about
                # meaningful layer property names in my network classes), but
                # it'll do
                if module.layer.__class__.__name__ == "BatchNorm2d":
                    nn.init.normal_(module.weight, 1, 0.02)
                    nn.init.constant_(module.bias, 0)
                if module.layer.__class__.__name__ in ["Conv2d", "ConvTranspose2d", "Linear"]:
                    nn.init.normal_(module.weight, 0, 0.02)
        # OK, the instructor's solution used `isinstance` instead of
        # `.__class__.__name__`, which is more natural and I don't know why I
        # didn't think it
        #
        # (I had also missed that `Linear` gets the same inits as the trans/cis
        # convolutional layers)


# (I also set the slope on my discriminator LeakyReLUs to 0.2 to match the solution.)

# Now to write the training loop! ... tomorrow?
