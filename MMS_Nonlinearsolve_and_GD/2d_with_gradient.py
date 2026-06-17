import math
import numpy as np
from pathlib import Path

from firedrake import *
from firedrake.output import VTKFile


# ----------------------------------------------------
# Output directory = directory containing this .py file
# ----------------------------------------------------
try:
    CODE_DIR = Path(__file__).resolve().parent
except NameError:
    CODE_DIR = Path.cwd()


# ----------------------------------------------------
# Mesh
# ----------------------------------------------------
mesh = UnitSquareMesh(64, 64)


# ----------------------------------------------------
# Function space
#
# q = (q1, q2)
#
# Q = [[ q1,  q2],
#      [ q2, -q1]]
# ----------------------------------------------------
V = VectorFunctionSpace(mesh, "CG", 1, dim=2)
S = FunctionSpace(mesh, "CG", 1)

q = Function(V, name="q")
q_new = Function(V, name="q_new")

u = TrialFunction(V)
p = TestFunction(V)

x, y = SpatialCoordinate(mesh)


# ----------------------------------------------------
# Parameters
# ----------------------------------------------------
s0 = Constant(0.7)

l1 = Constant(1.0)
eta = Constant(0.1)

a2 = Constant(7.5)
a3 = Constant(61.0)
a4 = Constant(66.52)

eps = Constant(1.0e-4)

tau_value = 1.0e-5
tau = Constant(tau_value)

max_iter = 20000
tol = 1.0e-10

save_every = 100


# ----------------------------------------------------
# q-vector -> strict traceless Q-tensor
# ----------------------------------------------------
def Q_tensor(v):
    return as_tensor([
        [v[0],  v[1]],
        [v[1], -v[0]],
    ])


# ----------------------------------------------------
# Manufactured exact solution
# ----------------------------------------------------
X = x - 0.5
Y = y - 0.5
r2 = X**2 + Y**2 + eps

q_exact = as_vector([
    (s0 / 2.0) * (X**2 - Y**2) / r2,
    s0 * X * Y / r2,
])

Q_exact = Q_tensor(q_exact)


# ----------------------------------------------------
# Manufactured forcing
#
# -l1 ΔQ + 1/eta^2(-a2 Q - a3 Q^2 + a4 |Q|^2 Q) = f
# ----------------------------------------------------
f_tensor = (
    -l1 * div(grad(Q_exact))
    + (1.0 / eta**2) * (
        -a2 * Q_exact
        -a3 * dot(Q_exact, Q_exact)
        + a4 * inner(Q_exact, Q_exact) * Q_exact
    )
)


# ----------------------------------------------------
# Boundary condition and initial guess
# ----------------------------------------------------
bc = DirichletBC(V, q_exact, "on_boundary")

q.interpolate(as_vector([
    0.05 * X,
    0.05 * Y,
]))

bc.apply(q)


# ----------------------------------------------------
# Semi-implicit gradient descent form
# ----------------------------------------------------
Q_trial = Q_tensor(u)
Q_old = Q_tensor(q)
P = Q_tensor(p)

bulk_old = (
    -a2 * Q_old
    -a3 * dot(Q_old, Q_old)
    + a4 * inner(Q_old, Q_old) * Q_old
)

A = (
    inner(Q_trial / tau, P) * dx
    + l1 * inner(grad(Q_trial), grad(P)) * dx
)

L = (
    inner(Q_old / tau, P) * dx
    - (1.0 / eta**2) * inner(bulk_old, P) * dx
    + inner(f_tensor, P) * dx
)

problem = LinearVariationalProblem(A, L, q_new, bcs=[bc])

solver = LinearVariationalSolver(
    problem,
    solver_parameters={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
)


# ----------------------------------------------------
# Output functions
# ----------------------------------------------------
Q00 = Function(S, name="Q00")
Q01 = Function(S, name="Q01")
Q10 = Function(S, name="Q10")
Q11 = Function(S, name="Q11")

W = VectorFunctionSpace(mesh, "CG", 1, dim=3)
director = Function(W, name="director")

q_out = Function(V, name="q")

Q_file = VTKFile(str(CODE_DIR / "Q_components_gradient_descent_time.pvd"))
director_file = VTKFile(str(CODE_DIR / "director_gradient_descent_time.pvd"))
q_file = VTKFile(str(CODE_DIR / "q_gradient_descent_time.pvd"))


def update_output_functions():
    Q_h = Q_tensor(q)

    Q00.project(Q_h[0, 0])
    Q01.project(Q_h[0, 1])
    Q10.project(Q_h[1, 0])
    Q11.project(Q_h[1, 1])

    q_out.assign(q)

    q_vals = q.dat.data_ro
    d_vals = director.dat.data

    for node in range(len(q_vals)):
        q1 = q_vals[node, 0]
        q2 = q_vals[node, 1]

        Qmat = np.array([
            [q1,  q2],
            [q2, -q1],
        ])

        eigvals, eigvecs = np.linalg.eigh(Qmat)
        n = eigvecs[:, np.argmax(eigvals)]

        d_vals[node, 0] = n[0]
        d_vals[node, 1] = n[1]
        d_vals[node, 2] = 0.0


def write_frame(k):
    t = k * tau_value

    update_output_functions()

    Q_file.write(Q00, Q01, Q10, Q11, time=t)
    director_file.write(director, time=t)
    q_file.write(q_out, time=t)


# ----------------------------------------------------
# Save initial frame
# ----------------------------------------------------
write_frame(0)


# ----------------------------------------------------
# Gradient descent loop
# ----------------------------------------------------
for k in range(1, max_iter + 1):

    solver.solve()

    update_norm = math.sqrt(float(assemble(
        inner(Q_tensor(q_new) - Q_tensor(q),
              Q_tensor(q_new) - Q_tensor(q)) * dx
    )))

    q.assign(q_new)
    bc.apply(q)

    if k % save_every == 0:
        write_frame(k)

    if k % 100 == 0:
        error_L2 = math.sqrt(float(assemble(
            inner(Q_tensor(q) - Q_exact, Q_tensor(q) - Q_exact) * dx
        )))

        print(
            f"iter = {k:5d}, "
            f"time = {k * tau_value:.6e}, "
            f"update = {update_norm:.3e}, "
            f"L2 error = {error_L2:.3e}"
        )

    if update_norm < tol:
        print(f"Converged at iter = {k}, update = {update_norm:.3e}")
        write_frame(k)
        break


print(
    "Saved in:\n"
    f"  {CODE_DIR}\n\n"
    "Files:\n"
    "  Q_components_gradient_descent_time.pvd\n"
    "  director_gradient_descent_time.pvd\n"
    "  q_gradient_descent_time.pvd"
)