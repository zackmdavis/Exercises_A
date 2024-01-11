import torch
import einops
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
    def forward(self, x):
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


def pad2d(x, left, right, top, bottom, pad_value):
    '''Return a new tensor with padding applied to the edges.
    x: shape (batch, in_channels, height, width), dtype float32
    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    '''
    B, C, H, W = x.shape
    output = x.new_full(size=(B, C, top + H + bottom, left + W + right), fill_value=pad_value)
    output[..., top : top + H, left : left + W] = x
    return output



def fractional_stride_2d(x, stride_h, stride_w):
    '''
    Same as fractional_stride_1d, except we apply it along the last 2 dims of x (width and height).
    '''
    batch, in_channels, height, width = x.shape
    width_new = width + (stride_w - 1) * (width - 1)
    height_new = height + (stride_h - 1) * (height - 1)
    x_new_shape = (batch, in_channels, height_new, width_new)

    # Create an empty array to store the spaced version of x in.
    x_new = torch.zeros(size=x_new_shape, dtype=x.dtype, device=x.device)

    x_new[..., ::stride_h, ::stride_w] = x

    return x_new



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
        ic_2, oc, kernel_height, kernel_width = self.weight.shape
        assert ic == ic_2, f"in_channels for x and weights don't match up. Shapes are {x.shape}, {self.weight.shape}."

        # Apply spacing
        x_spaced_out = fractional_stride_2d(x, stride_h, stride_w)

        # Apply modification (which is controlled by the padding parameter)
        pad_h_actual = kernel_height - 1 - padding_h
        pad_w_actual = kernel_width - 1 - padding_w
        assert min(pad_h_actual, pad_w_actual) >= 0, "total amount padded should be positive"
        x_mod = pad2d(x_spaced_out, left=pad_w_actual, right=pad_w_actual, top=pad_h_actual, bottom=pad_h_actual, pad_value=0)

        # Modify weights
        weights_mod = einops.rearrange(self.weight.flip(-1, -2), "i o h w -> o i h w")

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

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def get_dataset(dataset, train=True):
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

# Now to write the training loop!


import time
import numpy as np
from PIL import Image

class DCGANTrainer:
    def __init__(
            self,
            latent_dim_size=100,
            hidden_channels=[128, 256, 512],
            dataset="CELEB",
            batch_size=16,
            epochs=3,
            lr=0.0002,
            betas=(0.5, 0.999),
            seconds_between_evals=60,
    ):
        self.latent_dim_size = latent_dim_size
        self.hidden_channels = hidden_channels
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.betas = betas
        self.seconds_between_evals = seconds_between_evals

        self.criterion = nn.BCELoss()

        self.trainset = get_dataset(self.dataset)
        self.trainloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)

        batch, img_channels, img_height, img_width = next(iter(self.trainloader))[0].shape
        assert img_height == img_width

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DCGAN(
            self.latent_dim_size,
            img_height,
            img_channels,
            self.hidden_channels,
        ).to(self.device).train()

        self.generator_optimizer = torch.optim.Adam(
            self.model.generator.parameters(), lr=self.lr, betas=self.betas, maximize=True
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self.model.discriminator.parameters(), lr=self.lr, betas=self.betas, maximize=True
        )


    def training_step_discriminator(self, real, fake):
        # So what does a training step look like? We're trying to maximize
        # log(D(x)) + log(1-D(G(z))), which entails calculating it, then the
        # "invoke backprop, step the optimizer, zero the gradients" routine.
        #
        # I drafted the code that makes sense to me, but it's probably wrong
        # (`1-fake_p` is likely a type error, if the subtrahend needs to be a
        # gradient-bearing PyTorch object rather than an int, and presumably we
        # should be using the `self.criterion` set in the skeleton.
        real_p = self.model.discriminator(real)
        fake_p = self.model.discriminator(fake)
        u = (torch.log(real_p) + torch.log(1-fake_p)).sum()
        u.backward()
        self.discriminator_optimizer.step()
        self.discriminator_optimizer.zero_grad()
        return u

    def training_step_generator(self, fake):
        p = self.model.discriminator(fake)
        u = torch.log(p).sum()
        u.backward()
        self.generator_optimizer.step()
        self.generator_optimizer.zero_grad()
        return u

    @torch.inference_mode()
    def evaluate(self):
        self.model.generator.eval()
        latent_noise = torch.randn(4, self.latent_dim_size).to(self.device)

        raw_samples = self.model.generator(latent_noise)
        # thanks to GPT-4 for image-saving snippet
        samples = (raw_samples - raw_samples.min()) / (raw_samples.max() - raw_samples.min())
        samples = (samples * 255).byte()
        image_array = np.transpose(samples.cpu().numpy(), (0, 2, 3, 1))
        # Save each image in the batch
        for i in range(image_array.shape[0]):
            img = Image.fromarray(image_array[i], 'RGB')
            img.save("step{}_{}.png".format(self.step, i))
        print("saved evaluation images for step {}!".format(self.step))

        self.model.generator.train()


    # Following the skeleton code.
    def train(self):
        self.step = 0
        last_log_time = time.time()

        for epoch in range(self.epochs):
            progress_bar = tqdm(self.trainloader, total=len(self.trainloader))
            for real, _label in progress_bar:
                latent_noise = torch.randn(self.batch_size, self.latent_dim_size).to(self.device)
                real = real.to(self.device)
                fake = self.model.generator(latent_noise)

                # `detach` returns a new non-gradient-tracking tensor (sharing storage with the original)
                u_discriminator = self.training_step_discriminator(real, fake.detach())
                u_generator = self.training_step_generator(fake)

                self.step += real.shape[0]
                progress_bar.set_description(
                    "{}, disc_u={:.4f}, gen_u={:.4f}, examples_seen={}".format(
                        epoch, u_discriminator, u_generator, self.step
                    )
                )

                if time.time() - last_log_time > self.seconds_between_evals:
                    last_log_time = time.time()
                    self.evaluate()


# After some routine error-chasing, it looks like ... I don't have the hardware?
#
# `torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB
# (GPU 0; 3.81 GiB total capacity; 1.92 GiB already allocated; 1.67 GiB free;
# 2.02 GiB reserved in total by PyTorch) If reserved memory is >> allocated
# memory try setting max_split_size_mb to avoid fragmentation.  See
# documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF`
#
# Reducing batch size from 64 to 16 seems to have circumvented the memory
# issue. Next error was `RuntimeError: grad can be implicitly created only for
# scalar outputs.` GPT-4 suggests summing or averaging over the batch, so I
# slap a `.mean()` on my `u` expressions. (Oh, but maybe I should use `sum()` to
# try training faster?)
#
# And we're training!! I'm not sure whether these u numbers (I'm not calling
# them "loss" if we're doing gradient ascent) means we're on track; I need to
# implement `evaluate` to show the images.
#
# The solutions are using Weights and Biases (wandb), which section I skipped
# because signing up for an account on a third-party site is annoying—I assume
# I can just write the images locally.

# Preliminary results looking OK!



if __name__ == "__main__":
    DCGANTrainer().train()
