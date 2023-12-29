import functools
import math
import torch
import torch.nn.functional
from torch import nn

class ReLU(nn.Module):
    def forward(self, x):
        # This is supposed to apply ReLU elementwise ... is there a more
        # efficient vectorizey way to do this rather than this Python list
        # comprehension?
        return torch.maximum(x, torch.tensor(0.0))

# Trying out doing exercises in online colab; the fact that it has the tests already set up is convenient ...

# I tried `return self.weight @ x + self.bias` as my `forward` implementation
# for Linear, and I'm getting `RuntimeError: mat1 and mat2 shapes cannot be
# multiplied (64x512 and 10x512)`.

# OK, it looks like you're supposed to transpose the weight matrix (I was about
# to ask GPT-4, but I didn't)

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        super().__init__()
        bound = 1/math.sqrt(in_features)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features).uniform_(-bound, bound))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features).uniform_(-bound, bound))
        else:
            self.bias = None

    def forward(self, x):
        result = x @ self.weight.T
        if self.bias is not None:
            result += self.bias
        return result


# Exercise to implement Flatten—we're told that we can use torch.reshape. So,
# it should just be a matter of finding the new shape (where the merged layer
# is a product of the mergees).

# (This one took way too long to do, due to Coding Incompetence. Instructor's solution is similar.)

class Flatten(nn.Module):
    def __init__(self, start_dim=1, end_dim=-1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        true_end = len(input_.shape) - 1 if self.end_dim == -1 else self.end_dim

        start = input_.shape[:self.start_dim]
        to_merge = input_.shape[self.start_dim:true_end+1]
        tail = input_.shape[true_end+1:]

        if to_merge:
            merged = functools.reduce(lambda a, b: a*b, to_merge)
            new_shape = start + (merged,) + tail
            return input_.reshape(new_shape)
        else:
            return input_

x = torch.arange(24).reshape((2, 3, 4))
assert Flatten(start_dim=0)(x).shape == (24,), Flatten(start_dim=0)(x).shape
assert Flatten(start_dim=1)(x).shape == (2, 12), Flatten(start_dim=1)(x).shape
assert Flatten(start_dim=0, end_dim=1)(x).shape == (6, 4), Flatten(start_dim=0, end_dim=1)(x).shape


# We are asked: "can you see what makes logits non-unique (i.e. why any given set of probabilities might correspond to several different possible sets of logits)?"
#
# I reply (without taking too long to think): normalization is not injective.
# Instructor says: logits are translation invariant: you can add a constant to all of them.


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_):
        super().__init__()
        self.flatten = nn.Flatten()
        self.first_linear = nn.Linear(28**2, 100)
        self.relu = nn.ReLU()
        self.second_linear = nn.Linear(100, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.first_linear(x)
        x = self.relu(x)
        x = self.second_linear(x)
        return x

# I had initially wrapped the `nn.Linear`s with `nn.Parameter`, but the tests
# errored—the `Parameter`-ness is presumably already taken care of inside of
# the sub-`Module`.


# We are asked: "can you explain why we include a data normalization function in torchvision.transforms?"
#
# I reply: presumably the normalized data is better for the NN to learn
# from?—that's not really an answer, though, as long as the un-normalized
# values are in range for a float.
#
# In addition to "numerical issues" (I got that one), the instructor's answer
# says that we might happen to get stuck in a flat region of the unnormalized
# loss landscape.

# When I click the training cell in the collab, it finds a typo bug in my
# `forward` method, which makes me think that the tests in the colab were not
# very exhaustive! ... in fact, I can't even find the tests in the repo? You'd
# expect that to have caused an error in Colab.

# Exercise to add a validation loop done in Colab. Code was—

# with t.inference_mode():
#     tested = 0
#     hits = 0
#     for imgs, labels in mnist_testloader:
#       imgs = imgs.to(device)
#       labels = labels.to(device)
#       for img, label in zip(imgs, labels):
#           prediction = model(img)
#           if prediction.argmax() == label:
#               hits += 1
#           tested += 1
#     print("accuracy: {}%".format(100 * hits/tested))

# Questions about convolutions— (I watched the 3b1b video last night.)
# We are asked: "Why would convolutional layers be less likely to overfit data
# than standard linear (fully connected) layers?"
#
# I reply (hesitantly): the CNN's inductive bias fits e.g. image data where
# nearby pixels are meaningfully related? You can't accidentally overfit to
# spurious correlations between distant pixels if your architecture isn't
# considering them.
#
# Instructor's answer (similar idea, arguably, but simpler): fewer weights are
# learned!
#
# We are asked: "Suppose you fixed some random permutation of the pixels in an
# image, and applied this to all images in your dataset, before training a
# convolutional neural network for classifying images. Do you expect this to be
# less effective, or equally effective?"
#
# I reply: less effective, because permuting the pixels ruins the property of
# nearby pixels being related by the kernel.
#
# We are asked: "If you have a 28x28 image, and you apply a 3x3 convolution
# with stride 1, padding 1, what shape will the output be?"
#
# I reply: the padding lets us position a 3x3 window centered at the edge, and
# the stride means we're covering every position, so the output should also be
# 28x28. (Instructor's answer affirms.)

# Implementing Conv2d ... the shape and initialization of the weights is
# straightforward enough from the doc page, but I'm not sure how to apply
# `conv2d` ... but I shouldn't overthink it. (We're importing it; implementing
# it is a bonus exercise later.) That doc page says it takes `input` and
# `weight` as args.

# Running the tests, I get `AssertionError: The values for attribute 'shape' do
# not match: torch.Size([7, 9, 65, 259]) != torch.Size([7, 9, 34, 131]).`
# ... oh! I'm not handling the `stride` or `padding` args! Probably I can just
# store them in `__init__` and pass them on to `conv2d`?

class Conv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0
    ):
        super().__init__()
        bound = math.sqrt(1/(in_channels * kernel_size**2))
        self.weight = nn.Parameter(
            torch.FloatTensor(out_channels, in_channels, kernel_size, kernel_size).uniform_(-bound, bound)
        )
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(x, self.weight, stride=self.stride, padding=self.padding)

# Instructor's solution calculated the initialization differently, but I think
# my version (with `_uniform`) is a lot more readable than arithmetically
# manipulating `torch.rand`.

# MaxPool2D; the "help, I'm confused about what to do" fold-out says this is
# just a pass-through. Trivial.

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=1):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.max_pool2d(
            x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        )

    def extra_repr(self) -> str:
        return "kernel_size={}, stride={}, padding={}".format(self.kernel_size, self.stride, self.padding)

# Implementing BatchNorm2d is hard ... I don't feel like it's been adequately
# explained what it's supposed to do? Or I guess it has—
#
# If training:
# For each batch and each channel, normalize mean/value over the width/height, for this batch.
# Update `running_mean` and `running_var` using momentum (as specified in docs).
#
# If eval:
# Normalize using the running-mean and running-var stored earlier.
#
# But then I'm confused: what are the weights for? Aren't there supposed to be
# learnable weights? The doc page says, "γ and β are learnable parameter
# vectors of size C (where C is the input size). By default, the elements of γ
# are set to 1 and the elements of β are set to 0"—which makes it sound like
# what the Colab calls `running_mean` and `running_var` are the
# weights. They're just not being optimized with gradients ...?

# This is sufficiently confusing that I'm going to look at the solution ...
# I'm definitely going to need to come back for a second pass at this.

# If I'm so pathetic as to look at the solution, let me at least annotate it carefully—
class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # So the weight/bias parameters are apparently different from the
        # buffer we use to track running mean/var
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_variance", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0))

    def forward(self, x):
        # x (batch, channels, height, width)

        if self.training:
            # Instructor comment says: "Using keepdim=True so we don't have to
            # worry about broadasting them with x at the end". Meaning, keepdim
            # gives us the channel averages in the shape of [1, 3, 1, 1]
            # (rather than [3]), so that it auto-broadcasts when we say
            # `x − mean` later.
            num_channels = len(x[0])
            assert len(x.shape) == 4, x.shape
            mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
            assert mean.shape == (1, x.shape[1], 1, 1)
            variance = torch.var(x, dim=(0, 2, 3), unbiased=False, keepdim=True)
            assert variance.shape == (1, x.shape[1], 1, 1)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
            self.num_batches_tracked += 1
        else:
            # x could just be (channels, height, width) here apparently??
            #
            # More broastcast shaping; the instructor's solution uses `einops`,
            # which I continue to bounce off of because I'm retarded
            mean = self.running_mean.reshape(1, len(self.running_mean), 1, 1)
            variance = self.running_variance.reshape(1, len(self.running_variance), 1, 1)

        # more broadcast shaping, it seems!
        weight = self.weight.reshape(1, len(self.weight), 1, 1)
        bias = self.bias.reshape(1, len(self.bias), 1, 1)

        # We normalize, but there's a learnable scale/shift that will get hit
        # by gradients during backprop. (I guess some scalings decrease loss
        # more than others?)
        return ((x - mean) / torch.sqrt(variance + self.eps)) * weight + bias


class AveragePool(nn.Module):
    def forward(self, x):
        return torch.mean(x, dim=(2, 3))


# Implementing Resnet looks intimidating, but we need to do it in order to
# become stronger! (Also, it shouldn't rationally be intimidating—it's just
# plugging lego bricks together, like all modern programming.)


# I missed a couple kwargs at first, but otherwise this looks like the instructor's solution.
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, first_stride=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.left = nn.Sequential(
            Conv2d(in_features, out_features, 3, stride=first_stride, padding=1),
            BatchNorm2d(out_features),
            ReLU(),
            Conv2d(out_features, out_features, 3, stride=1, padding=1),
            BatchNorm2d(out_features),
        )
        if first_stride > 1:
            self.right = nn.Sequential(
                # changed from `padding=1` while debugging
                Conv2d(in_features, out_features, 1, stride=first_stride, padding=0),
                BatchNorm2d(out_features),
            )
        else:
            self.right = nn.Identity()

        self.final_relu = ReLU()

    def forward(self, x):
        left = self.left(x)
        right = self.right(x)
        return self.final_relu(left + right)


# This one is also easy.
class BlockGroup(nn.Module):
    def __init__(self, n_blocks, in_features, out_features, first_stride=1):
        super().__init__()
        self.blocks = nn.Sequential(
            *(
                [ResidualBlock(in_features, out_features, first_stride=first_stride)] +
                [ResidualBlock(out_features, out_features) for _ in range(n_blocks - 1)]
            )
        )

    def forward(self, x):
        return self.blocks(x)


# Corrected a few mistakes from instructor's solution, but overall mostly straightforward.
#
# There are a few more errors when I try to load the canonical weights.
#
# RuntimeError: Error(s) in loading state_dict for ResNet34:
# 	size mismatch for block_groups.1.blocks.0.right.0.weight: copying a param with shape torch.Size([128, 64, 1, 1]) from checkpoint, the shape in current model is torch.Size([128, 64, 3, 3]).
# 	size mismatch for block_groups.2.blocks.0.right.0.weight: copying a param with shape torch.Size([256, 128, 1, 1]) from checkpoint, the shape in current model is torch.Size([256, 128, 3, 3]).
# 	size mismatch for block_groups.3.blocks.0.right.0.weight: copying a param with shape torch.Size([512, 256, 1, 1]) from checkpoint, the shape in current model is torch.Size([512, 256, 3, 3]).


class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()
        in_features_per_group = [64] + out_features_per_group[:3]

        self.early_layers = nn.Sequential(
            # Kind of confused by in-channels=3, out-channels=64? Are three
            # "colors" being processed into a significantly larger number of
            # "features"?
            Conv2d(3, 64, 7, stride=2, padding=3),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(3, stride=2),
        )
        self.block_groups = nn.Sequential(*[
            BlockGroup(n_blocks, in_features, out_features, first_stride)
            for n_blocks, in_features, out_features, first_stride
            in zip(n_blocks_per_group, in_features_per_group, out_features_per_group, first_strides_per_group)
        ])
        self.later_layers = nn.Sequential(
            AveragePool(),
            Flatten(),
            Linear(512, n_classes),
        )

    def forward(self, x):
        return self.later_layers(self.block_groups(self.early_layers(x)))


# Then the course gives us some code to load the canonical RestNet34 weights
# into the code we just wrote and do some predictions.

my_resnet = ResNet34()

def copy_weights(my_resnet, pretrained_resnet):
    my_dict = my_resnet.state_dict()
    pretrained_dict = pretrained_resnet.state_dict()
    assert len(my_dict) == len(pretrained_dict), "Mismatching state dictionaries."

    state_dict_to_load = {
        my_key: pretrained_value
        for (my_key, my_value), (pretrained_key, pretrained_value) in zip(my_dict.items(), pretrained_dict.items())
    }

    my_resnet.load_state_dict(state_dict_to_load)

    return my_resnet

from torchvision import models
pretrained_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
my_resnet = copy_weights(my_resnet, pretrained_resnet)

import json
import os
import pathlib
from PIL import Image
from torchvision import transforms

chapter_directory = pathlib.Path(os.path.expanduser('~')) / "Code" / "ARENA_2.0" / "chapter0_fundamentals" / "exercises" / "part3_resnets"
image_directory = chapter_directory / "resnet_inputs"
imagenet_labelfile = chapter_directory / "imagenet_labels.json"

image_filenames = [
    "chimpanzee.jpg",
    "golden_retriever.jpg",
    "platypus.jpg",
    "frogs.jpg",
    "fireworks.jpg",
    "astronaut.jpg",
    "iguana.jpg",
    "volcano.jpg",
    "goofy.jpg",
    "dragonfly.jpg",
]

images = [Image.open(image_directory / filename) for filename in image_filenames]

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

prepared_images = torch.stack([IMAGENET_TRANSFORM(img) for img in images], dim=0)

assert prepared_images.shape == (len(images), 3, IMAGE_SIZE, IMAGE_SIZE)


with open(imagenet_labelfile) as f:
    imagenet_labels = list(json.load(f).values())

for filename, img in zip(image_filenames, prepared_images):
    my_resnet.eval()
    predictions = my_resnet(img)
    point_prediction = predictions.argmax().item()
    print(filename, "|" , imagenet_labels[point_prediction], "|", predictions[0][point_prediction].item())

# Output—
#
# chimpanzee.jpg | chimpanzee, chimp, Pan troglodytes | 19.332378387451172
# golden_retriever.jpg | golden retriever | 12.240525245666504
# platypus.jpg | platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus | 17.8731632232666
# frogs.jpg | toyshop | 9.811150550842285
# fireworks.jpg | fountain | 12.617237091064453
# astronaut.jpg | liner, ocean liner | 9.924823760986328
# iguana.jpg | common iguana, iguana, Iguana iguana | 18.734634399414062
# volcano.jpg | volcano | 23.102157592773438
# goofy.jpg | binoculars, field glasses, opera glasses | 9.237372398376465
# dragonfly.jpg | banana | 7.265051364898682
