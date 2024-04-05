# I'm sick of getting tripped up on trivial einops bullshit!!
#
# Let's review https://ajcr.net/Basic-guide-to-einsum/ and
# https://rockt.github.io/2018/04/30/einsum super-carefully, line-by-line, and
# then I'll be ready to go back to the main ARENA course.

# The latter also links to https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/

# Following https://ajcr.net/Basic-guide-to-einsum/—

a = torch.tensor([0, 1, 2])

b = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])

# Suppose we want to muliply elementwise (broadcasting appropriately) and then sum along the rows.

# `a * b` doesn't work: `RuntimeError: The size of tensor a (3) must match the
# size of tensor b (4) at non-singleton dimension `

# But `a.view(-1, 1) * b` reshapes into a column bector (`view` reshapes so the
# second dimension is 1, and the -1 is a infer-appropriately wildcard).

# In [26]: a.view(-1, 1) * b
# Out[26]:
# tensor([[ 0,  0,  0,  0],
#         [ 4,  5,  6,  7],
#         [16, 18, 20, 22]])

# In [30]: (a.view(-1, 1) * b).sum(dim=1)
# Out[30]: tensor([ 0, 22, 76])

# But equivalently, we have—

# In [31]: einops.einsum(a, b, 'i, i j -> i')
# Out[31]: tensor([ 0, 22, 76])

# Note, the prompt was elementwise multiplication—maybe that's why I was so
# confused in my ARENA study, because I was expecting it to be matrix
# multiplication? They're not unrelated because matrix multiplication is built
# out of elementwise multiplications and sums: the ijth entry is the sum of the
# elementwise product of the ith row of the first and the jth column of the
# second.

# That is, matrix multiplication is 'i j, j k -> ik'.

# These are the rules—
# • Repeating labels between input arrays means values along those dimensions
#   are multiplied (elementwise). Thus, labels must represent the same length.
# • Omitting a label from the output means summing along that axis.
# • Unsummed axes can appear in any order in the output.

# So, for 1D arrays of the same size, 'i, i -> i' is elementwise
# multiplication, and 'i, i ->' is the dot product.

# NumPy's einsum (which is what the post I'm reading is actually about) is
# significantly different from einops: 'ii->i' in np.einsum can extract a
# diagonal from a matrix, but I'm not seeing the equivalent in einops.einsum??

# For 2D matrices:
# 'i j, j k -> i k' is matrix multiplication, but
# 'i j, i j -> i j' is elementwise multiplication

# And that explains why I was so confused! I was imagining that
# '... batch d_model, batch d_model -> ...' was a batched matrix product.
#
# Why did I think that?

# Let's look at these other blog posts and then go back to the ARENA course.

# "einsum() is then easily expressed as a deeply-nested set of for-loops"
# Now you're speaking my language!!

# It's a for-loop structure where the "free indices" (that appear to the right
# of the arrow) go first, followed by the indices that get summed over.

# For the matrix multiplication example, 'i k, k j → i j', it's
#
# for i in range(Ni):
#     for j in range(Nj):
#         total = 0
#         for k in range(Nk):
#             total += A[i, k] * B[k, j]
#         C[i, j] = total

# There are four axes, but three `for`-loops, because k is covering the rows of
# one and the columns of the other.

# Free indices range over the output, and guide the computation of every
# element of it.

# I might not actually want to allocate time to see it through, but it's
# tempting to try to dash off a quick implementation of the for-loop expansion
# as a Rust macro, to prove that I really understand it.

# As a simplifying assumption, assume there are just two tensors. It seems like
# the obvious game plan is to use an ordinary Rust function to parse the arrow
# expression, then only use macros for the for-loop expansion.

# Or—you know, there's probably a more flexible way to express the computation
# expressed by a dynamic number of nested for loops.
