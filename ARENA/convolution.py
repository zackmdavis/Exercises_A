# Intermediate Stride Exercises (from the §0.2 notebook)

# I expected the as-strided solution to be `torch.as_strided(mat,
# (mat.size(0),) (mat.size(1)+1,))` (with a little debugging help from GPT-4),
# but it fails the tests for zero-length matr—no, I needed to `sum()`, too. OK.

# Next is matrix–vector multiplication (using only `as_strided`, `sum`, and
# elementwise multiplication.

# Am I allowed ... loops? Or is that missing the point? (Implementing
# schoolbook matrix multiplication with loops would be missing the point of
# using strides.)
#
# Ax⃗ is "ith vector element of x⃗, times ith column of A", summed ...

# "Hint 1" suggests first trying to create a matrix [i, j] = mat[i, j] ×
# vector[j] and then summing over the second dimension.

# The sticking point for me is, how does multiplication interact with `as_strided`?

# "Hint 2" says to use strides to create `vec_expanded`, then use elementwise
# multiplication. OK, that makes sense.

# In [20]: v.as_strided((3, 3), (0, 1))
# Out[20]:
# tensor([[1., 2., 3.],
#         [1., 2., 3.],
#         [1., 2., 3.]])


# And `sum` takes dim args.

# `return (mat * vec.as_strided(mat.shape, (0,1))).sum(1)` passes the first
# test but not the second. There's a hint about that.

# It says: "It's possible that the input matrices you recieve could themselves be the
# output of an as_strided operation, so that they're represented in memory in a
# non-contiguous way. Make sure that your as_stridedoperation is using the
# strides from the original input arrays". Tricky!

# `return (mat * vec.as_strided(mat.shape, (0,vec.stride()[0]))).sum(1)`
