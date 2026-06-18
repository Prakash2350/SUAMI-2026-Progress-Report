from pathlib import Path
import math
from firedrake import as_tensor, cos, sin,conditional,ge


CODE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = CODE_DIR / "mms_results"
VTK_DIR = OUT_DIR / "vtk"

problem_mode = "mms"

s0_value = 0.7

l1_value = 1.0
eta_value = 1.0

a2_value = 7.5
a3_value = 61.0
a4_value = 66.52

#eps_value = 1.0e-4
eps_value = 1.0e-2
omega_value = 0.1

homeotropic_w0 = 10.0
planar_w1 = 10.0
planar_w2 = 10.0


mesh_sizes = [16,32]

gradient_N = 5
gradient_time_step_value = 1.0e-4
gradient_max_iter = 100
gradient_tol = 1.0e-8
gradient_save_every = 10




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


def Q_exact_radial_ball(x, y, s0, eps):
    from firedrake import as_tensor

    X = x
    Y = y
    r2 = X**2 + Y**2 + eps

    q0 = (s0 / 2.0) * (X**2 - Y**2) / r2
    q1 = s0 * X * Y / r2

    return as_tensor([
        [q0, q1],
        [q1, -q0],
    ])

def Q_exact_tangential_disk(x, y, s0, eps):
    from firedrake import as_tensor

    r2 = x**2 + y**2 + eps

    # n = (y, -x) / sqrt(x^2 + y^2)
    q0 = (s0 / 2.0) * (y**2 - x**2) / r2
    q1 = -s0 * x * y / r2

    return as_tensor([
        [q0, q1],
        [q1, -q0],
    ])


Q_EXACT_CASES = {
    #"constant": Q_exact_constant,
    #"cos_pi_x": Q_exact_cos_pi_x,
    #"radial": Q_exact_radial,
    "disk_radial":Q_exact_radial_ball,
    "disk_tangential":Q_exact_tangential_disk
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
    "snes_rtol": 1.0e-8,
    "snes_atol": 1.0e-10,
    "snes_max_it": 20,
    "ksp_type": "gmres",
    "ksp_rtol": 1.0e-7,
    "ksp_atol": 1.0e-10,
    "ksp_max_it": 200,
    "pc_type": "bjacobi",
    "sub_pc_type": "ilu",
}
