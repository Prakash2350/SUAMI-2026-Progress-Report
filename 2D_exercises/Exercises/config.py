from pathlib import Path

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

mesh_sizes = [64, 128, 256]

gradient_N = 256
gradient_time_step_value = 1.0e-5
gradient_max_iter = 2000
gradient_tol = 1.0e-10
gradient_save_every = 50


def Q_exact_radial(x, y, s0, eps):
    from firedrake import as_tensor

    X = x - 0.5
    Y = y - 0.5
    r2 = X**2 + Y**2 + eps

    q0 = (s0 / 2.0) * (X**2 - Y**2) / r2
    q1 = s0 * X * Y / r2

    return as_tensor([
        [q0, q1],
        [q1, -q0],
    ])


Q_EXACT_CASES = {
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
