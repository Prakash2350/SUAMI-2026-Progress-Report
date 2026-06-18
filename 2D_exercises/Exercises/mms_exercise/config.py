from pathlib import Path
import math
from firedrake import as_tensor, cos, sin,conditional,ge


CODE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = CODE_DIR / "mms_results"
VTK_DIR = OUT_DIR / "vtk"

s0_value = 0.7

l1_value = 1.0
eta_value = 1.0

a2_value = 7.5
a3_value = 61.0
a4_value = 66.52

eps_value = 1.0e-4
omega_value = 0.1

homeotropic_w0 = 10.0
planar_w1 = 10.0
planar_w2 = 10.0

mesh_sizes = [32, 64, 128]

gradient_N = 5
gradient_time_step_value = 1.0e-5
gradient_max_iter = 2000
gradient_tol = 1.0e-10
gradient_save_every = 50


def Q_exact_constant(x, y, s0, eps):
    # n = (1, 0)
    q0 = conditional(ge(x, -1.0), 0.5 * s0, 0.5 * s0)
    q1 = 0.0 * q0

    return as_tensor([
        [q0, q1],
        [q1, -q0],
    ])


def Q_exact_cos_pi_x(x, y, s0, eps):
    # n = (cos(pi x), sin(pi x))

    q0 = 0.5 * s0 * cos(2.0 * math.pi * x)
    q1 = 0.5 * s0 * sin(2.0 * math.pi * x)

    return as_tensor([
        [q0, q1],
        [q1, -q0],
    ])


def Q_exact_radial(x, y, s0, eps):
    # n = (x - 0.5, y - 0.5) / sqrt((x - 0.5)^2 + (y - 0.5)^2)

    X = x - 0.5
    Y = y - 0.5
    r2 = X**2 + Y**2 + eps

    q0 = 0.5 * s0 * (X**2 - Y**2) / r2
    q1 = s0 * X * Y / r2

    return as_tensor([
        [q0, q1],
        [q1, -q0],
    ])


Q_EXACT_CASES = {
    "constant": Q_exact_constant,
    "cos_pi_x": Q_exact_cos_pi_x,
    "radial": Q_exact_radial,
}

DIRECT_SOLVER_PARAMETERS = {
    "snes_type": "newtonls",
    "snes_rtol": 1.0e-11,
    "snes_atol": 1.0e-12,
    "snes_max_it": 50,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

GRADIENT_SOLVER_PARAMETERS = {
    "snes_type": "newtonls",
    "snes_rtol": 1.0e-10,
    "snes_atol": 1.0e-12,
    "snes_max_it": 30,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
