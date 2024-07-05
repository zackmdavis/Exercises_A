# Following along with https://www.youtube.com/watch?v=VMj-3S1tku0

import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 3*x**2 - 4*x + 5

xs = np.arange(-5, 5, 0.25)
ys = f(xs)
plt.plot(xs, ys)
# plt.show()

# Everyone already knows that derivatives are ...

# And everyone already knows what partial derivatives are, so multi-input
# expressions shouldn't be confusing.

h = 0.0001

a = 2.
b = -3.
c = 10.

d1 = a*b + c
a += h
d2 = a*b + c

print((d2 - d1)/h) # -3.000000000010772

# d/da of a*b is b, simple
