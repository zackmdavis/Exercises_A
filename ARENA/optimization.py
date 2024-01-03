
# A bit of non-essential confusion doing the "Implement `opt_fn_with_sgd`"
# exercise in Colab: `pathological_curve_loss` takes two separate x and y
# Tensors as args, but `opt_fn_with_sgd` expects a (2,)-tensor. (And I expect
# that passing in T[0] and T[1] separately doesn't store the gradients
# properly.) But the solution does pass T[0] and T[1] separately, OK.

import torch

# We are asked: "do you think we should use in-place operations in our optimizer?"
# I reply: ... yes? (Save the memory; gradient descent is local; once you've
# stepped forward, where you were in the past doesn't matter for future
# results. Well, not quite: momentum makes the past matter, but that's not relevant.
#
# Instructor's answer is different and stronger: it's not about saving memory:
# it's that you want to see the change to the Parameter's storage.


# Implementing SGD!

# From the start, I'm confused why the skeleton code gives us `self.gs`, but
# also, individual `Parameter` objects have a `grad` property. Which one is the
# "actual" gradients? GPT-4 suggests that it's for storing velocities—oh, and
# the pseudocode on the PyTorch reference page has `g` variables.

# But how do I actually get the magical autodifferentiated gradient? The
# examples of `torch.optim.SGD` usage I've seen call `.backward()` on the
# result of computing the loss function (not inside the guts of SGD). Do I just
# assume that the grads are there?—that they get set whenever a `Parameter`
# gets passed to a function?

# Asking GPT-4 about leaf tensors ... "After you call .backward(), PyTorch
# populates the .grad attribute of leaf tensors with the gradient of the loss
# with respect to that tensor."

# `RuntimeError: grad can be implicitly created only for scalar outputs`—good to know.

# OK, this example really clarifies to me how the autodifferentiation magic works—
#
# In [16]: u = torch.nn.Parameter(torch.Tensor([0, 1, 2, 3, 4]))
#
# In [17]: def my_function(x):
#     ...:     return sum(x[i]**i for i in range(len(x)))
#     ...:
#
# In [18]: v = my_function(u)
#
# In [19]: v.backward()
#
# In [20]: u.grad
# Out[20]: tensor([  0.,   1.,   4.,  27., 256.])

# After some trivial fixes, tests are still failing numerically. I'd prefer to
# have one-shotted it, but I think it's time to look at the instructor's
# solution.
#
# ... I'm not immediately seeing where my solution differs. (I have multiple
# `for` loops where the instructor just has one "outer" loop, but that should
# be immaterial.) Do I need to run these locally side-by-side to track down the
# bug?

# The fact that not all of the elements mis-matched, and the absolute diff. is
# "only" 0.0068, makes me wonder if it's a subtle numerical difference in the
# implementations but my solution is still "morally" correct??
#
# Mismatched elements: 56 / 64 (87.5%)
# Greatest absolute difference: 0.006781339645385742 at index (15, 0) (up to 1e-05 allowed)
# Greatest relative difference: 0.5964073538942171 at index (13, 1) (up to 0 allowed)
#
# This theory is strengthened by the fact that when I run one optimizer step
# with both my solution and the instructor solution locally, and compare the
# `gs`es, everything matches.
#
# Comparing my original solution and the instructor's solution in detail,
# another maybe-superficial difference is that they didn't have a separate `b`
# tensor for the momentum; they just use `self.gs` the whole way through
# (unlike the PyTorch docs pseudocode) ...
#
# I eventually fixed my code, following the instructor's solution (last error
# was my indentation being wrong—oops!

class SGD:
    def __init__(self, params, lr, momentum=0.0, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.mu = momentum
        self.lmbda = weight_decay

        self.t = 0
        self.gs = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @torch.inference_mode()
    def step(self):
        for i, (g, p) in enumerate(zip(self.gs, self.params)):
            # We assume that someone has already called `.backward()`, which sets
            # `.grad` on the parameters.
            #
            # g_t ​← ∇θf_t(θ_{t−1})
            next_g = p.grad
            if self.lmbda != 0:
                # Then we apply weight decay.
                #
                # g_t ← g_t ​+ λθ_{t−1}
                next_g += self.lmbda * p
            if self.mu != 0:
                # Then we apply momentum.
                if self.t > 0:
                    # b_t ​← μb_{t−1} ​+ (1−τ)g_t
                    # (we're not doing τ-dampening in this exercise)
                    next_g += self.mu * g

            # θ_t​ ← θ_{t−1} − γg_t
            self.params[i] -= self.lr * next_g
            self.gs[i] = next_g

        self.t += 1

    def __repr__(self) -> str:
        return "SGD(lr={}, momentum={}, weight_decay={})".format(self.lr, self.mu, self.lmbda)

# The Colab says there are diminishing marginal returns from implementing
# RMSprop, Adam, and AdamW ...

# There's an exercise to rewrite the SGD exercise to allow parameter groups,
# which feels like a great way to atone for not getting that right the first time
# and needing to look at the instructor's solution.

# I implement, and after some trivial fixes (missing `self` arg, &c.) ... the tests
# fail numerically again (greatest absolute difference 0.00639).
#
# This is a bit disheartening. 😰 💔

class GroupCapableSGD:
    def __init__(self, params, lr=None, momentum=0.0, weight_decay=0.0):
        params = list(params)  # collect if it's a generator &c.
        self.param_groups = []
        if all(isinstance(param, torch.nn.Parameter) for param in params):
            assert lr is not None
            self.add_parameter_group(params, lr, momentum, weight_decay)
        else:
            for param_group in params:
                self.add_parameter_group(
                    param_group['params'],
                    param_group.get('lr', lr),
                    param_group.get('momentum', momentum),
                    param_group.get('weight_decay', weight_decay)
                )

        self.t = 0

    def add_parameter_group(self, params, lr, momentum, weight_decay):
        assert lr is not None

        for existing_group in self.param_groups:
            if params == existing_group['params']:
                raise ValueError("no parameters can appear in more than one group")

        self.param_groups.append(
            {
                'params': params,
                'lr': lr,
                'momentum': momentum,
                'weight_decay': weight_decay,
                'gs': [torch.zeros_like(p) for p in params]
             }
        )

    def zero_grad(self):
        for param_group in self.param_groups:
            for param in param_group['params']:
                param.grad = None

    @torch.inference_mode()
    def step(self):
        for param_group in self.param_groups:
            for i, (g, p) in enumerate(zip(param_group['gs'], param_group['params'])):
                next_g = p.grad
                if param_group['weight_decay'] != 0:
                    next_g += param_group['weight_decay'] * p
                if param_group['momentum'] != 0:
                    if self.t > 0:
                        next_g += param_group['momentum'] * g
                param_group['params'][i] -= param_group['lr'] * next_g
                param_group['gs'][i] = next_g
            self.t += 1
