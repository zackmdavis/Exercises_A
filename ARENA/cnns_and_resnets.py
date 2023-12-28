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
        return torch.Tensor([max(xi, 0) for xi in x])
    # Solutions suggest `torch.maximum(x, torch.tensor(0.0))`


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        print(start, to_merge, tail)

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


class AveragePool(nn.Module):
    def forward(self, x):
        return torch.mean(x, dim=(2, 3))


# Implementing Resnet looks intimidating!!
