import math
from dataclasses import dataclass
from typing import Any, Callable

from firedrake import *

from . import config


@dataclass
class MMSProblem:
    N: int
    anchoring: str
    exact_case_name: str
    mesh: Any
    V: Any
    q: Any
    p: Any
    q_exact: Any
    q_exact_fn: Any
    q_error: Any
    Q_exact: Any
    f_tensor: Any
    boundary_rhs: Any
    l1: Any
    eta: Any
    a2: Any
    a3: Any
    a4: Any
    Q_tensor: Callable[[Any], Any]
    surface_grad: Callable[[Any], Any]
    energy_form: Callable[[Any], Any]
    compute_errors: Callable[[Any], tuple[float, float]]

    @property
    def h(self):
        return 1.0 / self.N


def build_problem(N, anchoring, exact_case_name="radial"):
    if anchoring not in {"homeotropic", "planar"}:
        raise ValueError("anchoring must be 'homeotropic' or 'planar'")

    if exact_case_name not in config.Q_EXACT_CASES:
        available = ", ".join(config.Q_EXACT_CASES)
        raise ValueError(
            f"unknown exact_case_name {exact_case_name!r}; "
            f"available cases: {available}"
        )

    mesh = UnitSquareMesh(N, N)

    V = VectorFunctionSpace(mesh, "CG", 1, dim=2)
    q = Function(V, name="q")
    p = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    normal = FacetNormal(mesh)
    I2 = Identity(2)

    s0 = Constant(config.s0_value)

    l1 = Constant(config.l1_value)
    eta = Constant(config.eta_value)

    a2 = Constant(config.a2_value)
    a3 = Constant(config.a3_value)
    a4 = Constant(config.a4_value)

    eps = Constant(config.eps_value)
    omega = Constant(config.omega_value)

    if anchoring == "homeotropic":
        w0 = Constant(config.homeotropic_w0)
        w1 = Constant(0.0)
        w2 = Constant(0.0)
    else:
        w0 = Constant(0.0)
        w1 = Constant(config.planar_w1)
        w2 = Constant(config.planar_w2)

    def Q_tensor(v):
        return as_tensor([
            [v[0], v[1]],
            [v[1], -v[0]],
        ])

    def Q_tilde(Q):
        return Q + (s0 / 2.0) * I2

    def Q_perp(Q):
        Pi = I2 - outer(normal, normal)
        return dot(Pi, dot(Q, Pi))

    def Q_gamma_homeotropic():
        return s0 * (outer(normal, normal) - I2 / 2.0)

    def surface_w2_grad(Q):
        Qt = Q_tilde(Q)
        return (w2 / omega) * (inner(Qt, Qt) - s0**2) * Qt

    def surface_grad(Q):
        if anchoring == "homeotropic":
            return w0 * (Q - Q_gamma_homeotropic())

        return w1 * (Q - Q_perp(Q)) + surface_w2_grad(Q)

    def bulk_energy_density(Q):
        return (1.0 / eta**2) * (
            -0.5 * a2 * inner(Q, Q)
            - (a3 / 3.0) * tr(dot(dot(Q, Q), Q))
            + 0.25 * a4 * inner(Q, Q) ** 2
        )

    def surface_energy_density(Q):
        if anchoring == "homeotropic":
            A = Q - Q_gamma_homeotropic()
            return 0.5 * w0 * inner(A, A)

        Qt = Q_tilde(Q)
        A = Q - Q_perp(Q)
        return (
            0.5 * w1 * inner(A, A)
            + (w2 / (4.0 * omega)) * (inner(Qt, Qt) - s0**2) ** 2
        )

    Q_exact = config.Q_EXACT_CASES[exact_case_name](x, y, s0, eps)
    q_exact = as_vector([Q_exact[0, 0], Q_exact[0, 1]])

    bulk_exact = (
        -a2 * Q_exact
        - a3 * dot(Q_exact, Q_exact)
        + a4 * inner(Q_exact, Q_exact) * Q_exact
    )

    f_tensor = (
        -l1 * div(grad(Q_exact))
        + (1.0 / eta**2) * bulk_exact
    )

    ii, jj, kk = indices(3)
    dQdn_exact = as_tensor(
        Q_exact[ii, jj].dx(kk) * normal[kk],
        (ii, jj),
    )
    boundary_rhs = l1 * dQdn_exact + surface_grad(Q_exact)

    q_exact_fn = Function(V, name="q_exact")
    q_exact_fn.interpolate(q_exact)
    q_error = Function(V, name="q_error")

    def energy_form(Q):
        return (
            0.5 * l1 * inner(grad(Q), grad(Q)) * dx
            + bulk_energy_density(Q) * dx
            - inner(f_tensor, Q) * dx
            + surface_energy_density(Q) * ds
            - inner(boundary_rhs, Q) * ds
        )

    def compute_errors(q_func):
        Q_num = Q_tensor(q_func)
        Q_err = Q_num - Q_exact

        L2_err = math.sqrt(float(assemble(
            inner(Q_err, Q_err) * dx
        )))

        H1_err = math.sqrt(float(assemble(
            inner(Q_err, Q_err) * dx
            + inner(grad(Q_err), grad(Q_err)) * dx
        )))

        return L2_err, H1_err

    return MMSProblem(
        N=N,
        anchoring=anchoring,
        exact_case_name=exact_case_name,
        mesh=mesh,
        V=V,
        q=q,
        p=p,
        q_exact=q_exact,
        q_exact_fn=q_exact_fn,
        q_error=q_error,
        Q_exact=Q_exact,
        f_tensor=f_tensor,
        boundary_rhs=boundary_rhs,
        l1=l1,
        eta=eta,
        a2=a2,
        a3=a3,
        a4=a4,
        Q_tensor=Q_tensor,
        surface_grad=surface_grad,
        energy_form=energy_form,
        compute_errors=compute_errors,
    )
