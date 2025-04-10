"""Provides a PyTorch module that wraps a CasADi expression.

TODO (Jasper): Does JIT compilation speed up the module?
"""

import casadi as ca
import numpy as np
import torch
import torch.autograd as autograd
from acados_template import AcadosSimSolver


class AutogradCasadiFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx, f: ca.Function, jtimes: ca.Function, *inputs: torch.Tensor
    ) -> torch.Tensor:
        assert inputs[0].ndim == 2
        device = inputs[0].device
        dtype = inputs[0].dtype

        # assert that the inputs are 1D and batched assuming first dimension is batch size
        inputs = [input.detach().cpu().numpy().T for input in inputs]

        # save the jacobian and inputs for the backward pass
        ctx.inputs = inputs
        ctx.jtimes = jtimes

        result = f(*inputs).full()

        # convert numpy to tensor and transpose
        return torch.tensor(result, device=device, dtype=dtype).T

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        # convert grad_output to numpy
        device = grad_output.device
        dtype = grad_output.dtype
        grad_output = grad_output.detach().cpu().numpy().T

        # prepare inputs and remember the splits
        inputs = ctx.inputs
        sizes = [input.shape[0] for input in inputs]
        splits = np.cumsum(sizes)[:-1]
        inputs = np.concatenate(inputs, axis=0)

        # compute the jacobian
        grad_input = ctx.jtimes(inputs, grad_output).T
        # split the grads for the individual inputs
        grad_input = np.split(grad_input, splits, axis=1)
        # convert to tensor
        grad_input = [
            torch.tensor(grad, device=device, dtype=dtype) for grad in grad_input
        ]

        return (None, None, *grad_input)


class DynamicsSimFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        sim: AcadosSimSolver,
        x: torch.Tensor,
        u: torch.Tensor,
        p: torch.Tensor | None,
    ) -> torch.Tensor:
        device = x.device
        dtype = x.dtype
        batch_size = x.shape[0]
        xdim = x.shape[1]
        udim = u.shape[1]

        # assert that the inputs are 1D and batched assuming first dimension is batch size
        x = x.detach().cpu().numpy()
        u = u.detach().cpu().numpy()
        if p is not None:
            p = p.detach().cpu().numpy()

        x_next = np.zeros_like(x)
        Sx = np.zeros((batch_size, xdim, xdim))
        Su = np.zeros((batch_size, xdim, udim))

        for i in range(x.shape[0]):
            # TODO (Jasper): Replace with method simulate!
            sim.set("x", x[i, :])
            sim.set("u", u[i, :])
            if p is not None:
                sim.set("p", p[i, :])
            sim.solve()
            x_next[i, :] = sim.get("x")
            Sx[i, ...] = sim.get("Sx")
            Su[i, ...] = sim.get("Su")

        # save the jacobian for the backward pass
        ctx.Sx = Sx
        ctx.Su = Su

        return torch.tensor(x_next, device=device, dtype=dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        # convert grad_output to numpy
        device = grad_output.device
        dtype = grad_output.dtype

        # compute gradients of the inputs
        Sx = torch.tensor(ctx.Sx, device=device, dtype=dtype)
        Su = torch.tensor(ctx.Su, device=device, dtype=dtype)
        grad_x = torch.einsum("bkj,bk->bj", Sx, grad_output)
        grad_u = torch.einsum("bkj,bk->bj", Su, grad_output)

        # print(Sx[0])
        # print(grad_x[0])

        return (None, grad_x, grad_u, None)


