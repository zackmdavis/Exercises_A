import einops
import torch
import plotly.express
import plotly.graph_objects

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
    print(rays.shape)
    return rays


def make_rays_2d_instructor_solution(num_pixels_y, num_pixels_z, y_limit, z_limit):
    n_pixels = num_pixels_y * num_pixels_z
    ygrid = torch.linspace(-y_limit, y_limit, num_pixels_y)
    zgrid = torch.linspace(-z_limit, z_limit, num_pixels_z)
    rays = torch.zeros((n_pixels, 2, 3), dtype=torch.float32)
    rays[:, 1, 0] = 1
    rays[:, 1, 1] = einops.repeat(ygrid, "y -> (y z)", z=num_pixels_z)
    rays[:, 1, 2] = einops.repeat(zgrid, "z -> (y z)", y=num_pixels_y)
    print(rays.shape)
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

if __name__ == "__main__":
    make_rays_2d(10, 10, 0.3, 0.3)
    make_rays_2d_instructor_solution(10, 10, 0.3, 0.3)
