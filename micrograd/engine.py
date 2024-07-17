import random
import math


class Value:
    def __init__(self, value, _children=(), _operation='', label=''):
        self.value = value
        self._previous = set(_children)
        self._operation = _operation
        self._backward = lambda: None  # default doesn't do anything
        self.label = label
        self.gradient = 0

    def __repr__(self):
        return "<Value: {}>".format(self.value)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value + other.value, (self, other), '+')
        # Backward function for sum result.
        def _backward():
            # addition just passes the gradient through to both branches
            self.gradient += out.gradient
            other.gradient += out.gradient
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value * other.value, (self, other), '*')
        # For multiplication, d/dx(xy) = y, and d/dy(xy) = x.
        def _backward():
            self.gradient += other.value * out.gradient
            other.gradient += self.value * out.gradient
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.value**other, (self,), "^{}",format(other))

        def _backward():
             self.gradient = other * self.value ** (other-1) * out.gradient
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other**-1

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return self + (-other)

    def tanh(self):
        t = (math.exp(2*self.value) - 1)/(math.exp(2*self.value) + 1)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.gradient += (1 - t**2) * out.gradient
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.value < 0 else self.value)
        def _backward():
            self.gradient += (out.value > 0) * out.gradient
        self._backward = _backward
        return out

    def exp(self):
        x = self.value
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.gradient += out.value * out.gradient
        out._backward = _backward
        return out

    def backward(self):
        import pudb; pudb.set_trace()
        graph = []
        visited = set()
        def build_topological_order(v):
            if v not in visited:
                visited.add(v)
                for child in v._previous:
                    build_topological_order(child)
                graph.append(v)
        build_topological_order(self)

        self.gradient = 1.
        for v in reversed(graph):
            v._backward()

# The gradient of the loss with respect to itself, dL/dL, is one.

# To start, Karpathy hardcodes two computation graphs ending in a node L, and
# modifies one node in the second one with "+ h". The gradient with respect to
# that variable is (L2 - L1)/h.

# At a '+' node, the derivative just gets forwarded/split.

# In the first part of the lecture video, Karpathy manually demonstrates
# multiplying the local gradient with the "upstream" (?), but to do it
# automatically, we code up the "backwards" function for each operation type.

a = Value(2., label='a')
b = Value(-3., label='b')
c = Value(10., label='c')
d = a * b + c
d.label = 'd'

# In [11]: d.backward()
#
# In [12]: for v in [a, b, c, d]:
#     ...:     print(v.label, v.gradient)
#     ...:
# a -3.0
# b 2.0
# c 1.0
# d 1.0

# So, it works? If we have multiplication and addition, and do ReLU, will that
# be enough to build something ambitious like a transformer?? But I'm getting
# ahead of myself

# Karpathy discusses why our code so far breaks when you re-use variables. The
# solution is to accumulate gradients—in case you are wondering why
# `.zero_grad` is part of the magical PyTorch incantations.

# I would have reached for "just don't reuse variables".  But think of a
# DAG. Clearly, it's legit for parent nodes to have more than one child, and
# that's a reasonable way to put c := a*b and d := a + b in the same graph. But
# d.backward() shouldn't clobber the gradients.

# We can also do auto-wrapping.

# Karpathy contrasts to PyTorch. Here, we've been using scalar values, but
# PyTorch mostly uses tensors.

import torch

a_ = torch.tensor([2.], requires_grad=True)
b_ = torch.tensor([-3.], requires_grad=True)
c_ = torch.tensor([10.], requires_grad=True)
d_ = a_ * b_ + c_

# In [2]: d_.retain_grad()
#    ...: d_.backward()
#    ...: for v in [a_, b_, c_, d_]:
#    ...:     print(v.grad)
#    ...:

# [probably irrelevant UserWarning clipped —Ed.]
# tensor([-3.])
# tensor([2.])
# tensor([1.])
# tensor([1.])

# Karpathy says that now we're ready to build neural nets. Let's go!!

class Neuron:
    def __init__(self, input_count):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(input_count)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # w * x + b
        activation = sum(wi*xi for wi, xi in zip(self.weights, x)) + self.bias
        out = activation.relu()
        return out

# x = [2.0, 3.0]
# n = Neuron(2)
# print(n(x))

# Funny gotcha: the value here is zero half the time—because that's how ReLUs work. It's not a bug.

class Layer:
    def __init__(self, in_dimensionality, out_dimensionality):
        self.neurons = [Neuron(in_dimensionality) for _ in range(out_dimensionality)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs

# n = Layer(2, 3)
# print(n(x))

class MultiLayerPerceptron:
    def __init__(self, in_dimensionality, layer_dimensionalities):
        layer_sizes = [in_dimensionality] + layer_dimensionalities
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_dimensionalities))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

n = MultiLayerPerceptron(3, [4, 4, 1])

xs = [
    [2., 3., -1.],
    [3., -1, 0.5],
    [0.5, 1., 1.],
    [1., -1., -1.],
]

ys = [1., -1., -1., 1.]

ys_hat = [n(x)[0] for x in xs]

losses = [(y_true - y_out)**2 for y_true, y_out in zip(ys, ys_hat)]

# The reason PyTorch has `requires_grad` is because the full computation graph
# also includes gradients for the input data. But that's not very
# useful. (Unless we're doing adversarial attacks, we're not going to do SGD on
# the input.)

# ... my gradients don't seem to be updating when I call losses[0].backward().

# I think the code I have now might be insufficiently recursive? Dropping into
# `pudb`, I'm seeing the topologically sorted list having four values, but the
# full computational graph is much larger than that.

# But the code I have now does seem to match
# https://github.com/karpathy/micrograd/blob/c911406e5ace8742e5841a7e0df113ecb5d54685/micrograd/engine.py#L54-L70
# ?
