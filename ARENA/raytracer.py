import einops
import torch
import plotly.express
import plotly.graph_objects
from plotly.express import imshow

# https://arena3-chapter0-fundamentals.streamlit.app/[0.1]_Ray_Tracing

def render_lines(lines, bold_lines=torch.Tensor()):
    figure = plotly.graph_objects.Figure(layout={'showlegend': False, 'height': 600, 'width': 600})
    for lines, kwargs in [(lines, {}), (bold_lines, {'line_width': 5, 'line_color': "black"})]:
        for line in lines:
            x, y, z = line.T
            figure.add_scatter3d(x=x, y=y, z=z, mode="lines", **kwargs)
    figure.show()


def make_rays_1d(num_pixels, y_limit):
    # As a Python normie, I originally tried to do this with lists—
    #
    # rays = []
    # for i in range(num_pixels+1):
    #     rays.append([[0, 0, 0], [1, -y_limit + i*(2 * y_limit / num_pixels), 0]])
    # return rays
    #
    # But the instructor's solution uses Torch magic—
    rays = torch.zeros((num_pixels, 2, 3), dtype=torch.float32)
    torch.linspace(-y_limit, y_limit, num_pixels, out=rays[:, 1, 1])
    rays[:, 1, 0] = 1
    return rays


def intersect_ray_1d(ray, segment):
    # The linear algebra makes sense, but I'm not sure what's wrong with my
    # code. (I tried building the 3×2 matrix first, then added the [:2] when I
    # ran into the "must be batches of square matrices" error that the guide
    # warns about.)
    #
    # torch.linalg.solve(
    #     torch.cat((ray[1].reshape(-1, 1), (segment[0] - segment[1]).reshape(-1, 1)), dim=1)[:2],
    #     (segment[0] - ray[0]).reshape(-1, 1)[:2]
    # )
    #
    # The instructor's solution goes like this—
    ray = r[..., :2]
    segment = s[..., :2]

    O, D = r
    L_1, L_2 = s

    mat = t.stack([D, L_1 - L_2], dim=-1)
    vec = L_1 - O

    try:
        sol = t.linalg.solve(mat, vec)
    except:
        return False

    u = sol[0].item()
    v = sol[1].item()
    return (u >= 0.0) and (v >= 0.0) and (v <= 1.0)


# Exercise to implement `intersect_rays_1d` makes me feel very dumb. I think
# it's the vectorization that's getting me: the linear algebra makes sense;
# doing linear algebra in a `for` loop makes sense; but wrapping my head around
# a 4D tensor feels bad.


# This one should be easy—the analogy to `make_rays_1d` should just work—well,
# except that we need the `linspace` analogue of nested `for` loops ...
#
# I'm going to need to repeat the linspace operation multiple times to get them
# to Cartesian product with each other—oh! Is this the "broadcasting" thing
# that the page mentions as a tip?—maybe not.
def make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit):
    rays = torch.zeros((num_pixels_y * num_pixels_z, 2, 3), dtype=torch.float32)
    rays[:, 1, 1] = torch.linspace(-y_limit, y_limit, num_pixels_y).repeat(num_pixels_z)
    rays[:, 1, 2] = torch.linspace(-z_limit, z_limit, num_pixels_z).repeat(num_pixels_y)
    rays[:, 1, 0] = 1
    return rays


def make_rays_2d_instructor_solution(num_pixels_y, num_pixels_z, y_limit, z_limit):
    n_pixels = num_pixels_y * num_pixels_z
    ygrid = torch.linspace(-y_limit, y_limit, num_pixels_y)
    zgrid = torch.linspace(-z_limit, z_limit, num_pixels_z)
    rays = torch.zeros((n_pixels, 2, 3), dtype=torch.float32)
    rays[:, 1, 0] = 1
    rays[:, 1, 1] = einops.repeat(ygrid, "y -> (y z)", z=num_pixels_z)
    rays[:, 1, 2] = einops.repeat(zgrid, "z -> (y z)", y=num_pixels_y)
    return rays

# I get a (non-axis aligned) triangle, rather than the pyramid I was supposed to get ...
#
# But both my "solution" and the canonical one have shape [100, 2, 3], which
# makes my mistake obvious; I'm not doing the Cartesian product correctly: I'm
# doing the equivalent of (1, 1), (2, 2), &c., when the real answer is of
# course (1, 1), (1, 2) ... (2, 1), &c.
#
# I should understand, what is this "einops.repeat" incantation such that it's
# doing the Cartesian product thing correctly?
#
# It looks like `(n x)` meaning "repeat tensor x, n times" and `(x n)` meaning
# "repeat each element of tensor x n times" just ... need to be memorized?


# This should generalize from `intersect_ray_1d` and the presented matrix
# equation pretty straightforwardly?
#
# The instructor's answer had `dim=1` (which probably makes column vectors?)
# and I got the barycentric coördinate check wrong, but otherwise looking
# similar. My code revised—
def ray_intersects_triangle(origin, direction, a, b, c):
    system = torch.stack(-direction, b - a, c - a)
    resultant = o - a

    try:
        solution = torch.linalg.solve(system, resultant, dim=1)
    except RuntimeError:
        return False

    _s, u, v = solution
    _s = s.item()
    u = u.item()
    v = v.item()

    return u >= 0 and v >= 0 and u + v <= 1


# This one is supposed to generalize from the vectorized `intersect_rays_1d`
# that made me feel very stupid earlier.
#
# How do I set up the "batched" system? If I can figure out how the earlier
# exercise made the dimensions line up, I should be able to adapt that?
#
# The earlier exercise had D.shape = L_1.shape = (NR, NS, 2)
# So the LHS `stack([D, L_1 - L_2], dim=-1)` would have had shape ... (NR, NS, 2, 2)?
#
# The first two dimensions index over the N rays and the N segments, and the
# third and fourth dimensions have two slots each to actually store the ray and
# the segment?
#
# And the RHS would have had shape (NR, NS, 2). I don't understand why this is
# a legal matrix equation.
#
# GPT-4 and the man page
# (https://pytorch.org/docs/stable/generated/torch.linalg.solve.html) explain
# that "batch dimensions" are a thing. The last two dimensions on the left, and
# the last dimension on the right are the Ax⃗ = b⃗ that I'm familiar with, and
# any preceding dimensions are specifying versions of that calculation in
# parallel.

# My code looks generally pretty similar to the instructor's, but I'm getting
# nonsense answers, and I don't have the same linear system as them. :'( </3
#
# ...

def raytrace_triangle(rays, triangle):
    # rays (n, 2 origin/direction points, 3 dimensions)
    # triangle (3 points, 3 dimensions)
    n = rays.size(0)

    origins = rays[:, 0, :]
    directions = rays[:, 1, :]
    # I erroneously expected these to be (rays.shape(0), 1, 3), but snipping
    # out a dimension isn't the same thing as having a dimension with one slot
    # in it
    assert origins.shape == (n, 3)
    assert directions.shape == (n, 3)

    a = triangle[0]
    b = triangle[1]
    c = triangle[2]
    assert a.shape == (3,)
    assert b.shape == (3,)
    assert c.shape == (3,)

    b_less_a_system = (b - a).tile(n, 1)
    c_less_a_system = (c - a).tile(n, 1)
    assert b_less_a_system.shape == (225, 3), "b⃗ − a⃗ shape is {}".format(b_less_a_system.shape)
    assert c_less_a_system.shape == (225, 3), "c⃗ − a⃗ shape is {}".format(b_less_a_system.shape)
    system = torch.stack([-directions, b_less_a_system, c_less_a_system], dim=1)
    assert system.shape == (n, 3, 3), "system shape is {}" .format(system.shape)

    resultants = origins - a.tile(n, 1)
    assert resultants.shape == (n, 3)

    # I think we can disregard singular cases if we assume that the rays are
    # never perfectly skew to the plane

    print("my system:\n{}".format(system[:5]))
    print("my resultant:\n{}".format(resultants[:5]))

    solution = torch.linalg.solve(system, resultants)
    assert solution.shape == (n, 3)

    return torch.Tensor([u >= 0 and v >= 0 and u + v <= 1 for _s, u, v in solution])


def raytrace_triangle_instructor_solution(rays, triangle):
    NR = rays.size(0)

    # Triangle is [[Ax, Ay, Az], [Bx, By, Bz], [Cx, Cy, Cz]]
    A, B, C = einops.repeat(triangle, "pts dims -> pts NR dims", NR=NR)
    assert A.shape == (NR, 3)

    # Each element of `rays` is [[Ox, Oy, Oz], [Dx, Dy, Dz]]
    O, D = rays.unbind(dim=1)
    assert O.shape == (NR, 3)

    # Define matrix on left hand side of equation
    mat = torch.stack([- D, B - A, C - A], dim=-1)

    # Get boolean of where matrix is singular, and replace it with the identity in these positions
    # Note - this works because mat[is_singular] has shape (NR_where_singular, 3, 3), so we
    # can broadcast the identity matrix to that shape.
    dets = torch.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = torch.eye(3)

    # Define vector on the right hand side of equation
    vec = O - A

    print("instructor's system:\n{}".format(mat[:5]))
    print("instructor's resultant:\n{}".format(vec[:5]))

    # Solve eqns
    sol = torch.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=-1)

    # Return boolean of (matrix is nonsingular, and solution is in correct range implying intersection)
    return ((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)


if __name__ == "__main__":
    # instructor's demo code
    A = torch.tensor([1, 0.0, -0.5])
    B = torch.tensor([1, -0.5, 0.0])
    C = torch.tensor([1, 0.5, 0.5])
    num_pixels_y = num_pixels_z = 15
    y_limit = z_limit = 0.5

    # Plot triangle & rays
    test_triangle = torch.stack([A, B, C], dim=0)
    rays2d = make_rays_2d_instructor_solution(num_pixels_y, num_pixels_z, y_limit, z_limit)

    # Calculate and display intersections
    intersects = raytrace_triangle(rays2d, test_triangle)
    canon_intersects = raytrace_triangle_instructor_solution(rays2d, test_triangle)
