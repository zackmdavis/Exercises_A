import math


class Value:
    def __init__(self, value, _children=(), _operation='', label=''):
        self.value = value
        self._previous = set(_children)
        self._operation = _operation
        self.label = label
        self.gradient = gradient

    def __repr__(self):
        return "<Value: {}>".format(self.value)

    def __add__(self, other):
        return Value(self.value + other.value, (self, other), '+')

    def __mul__(self, other):
        return Value(self.value * other.value, (self, other), '*')

    def tanh(self):
        return Value((math.exp(2*self.value) - 1)/(math.exp(2*self.value) + 1), (self,), 'tanh')


a = Value(2., label='a')
b = Value(-3., label='b')
c = Value(10., label='c')
d = a * b + c

print(d)
print(d._previous)
print(d._operation)

# The gradient of the loss with respect to itself, dL/dL, is one.

# To start, Karpathy hardcodes two computation graphs ending in a node L, and
# modifies one node in the second one with "+ h". The gradient with respect to
# that variable is (L2 - L1)/h.

# At a '+' node, the derivative just gets forwarded/split.

# In the first part of the lecture video, Karpathy manually demonstrates
# multiplying the local gradient with the "upstream" (?), but to do it
# automatically, we code up the "backwards" function for each operation type.
