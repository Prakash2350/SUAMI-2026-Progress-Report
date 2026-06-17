import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from firedrake import *
from firedrake.output import VTKFile

import numpy as np


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
eta = Constant(1.0)

a2 = Constant(7.5)
a3 = Constant(61.0)
a4 = Constant(66.52)

eps = Constant(1.0e-4)

# Surface parameters
w0 = Constant(0.0)
w1 = Constant(10.0)
w2 = Constant(10.0)
omega = Constant(0.1)


# ----------------------------------------------------
# Convert q-vector into 2D traceless Q-tensor
# ----------------------------------------------------
def Q_tensor(v):
    return as_matrix([
        [v[0],  v[1]],
        [v[1], -v[0]]
    ])


# ----------------------------------------------------
# Exact manufactured solution
#
# n_exact = (X, Y) / sqrt(X^2 + Y^2 + eps)
#
# Q_exact = s0 (n otimes n - I/2)
# ----------------------------------------------------
X = x - 0.5
Y = y - 0.5
r2 = X**2 + Y**2 + eps

I = Identity(2)

Q_exact = s0 * (
    as_matrix([
        [X**2, X*Y],
        [X*Y, Y**2]
    ]) / r2
    - I/2
)

q_exact = as_vector([
    Q_exact[0, 0],
    Q_exact[0, 1]
])


# ----------------------------------------------------
# Manufactured volume forcing
#
# Strong form:
#
# -l1 Delta Q
# + 1/eta^2 (-a2 Q - a3 Q^2 + a4 |Q|^2 Q)
# = f
# ----------------------------------------------------
Q_exact_2 = Q_exact * Q_exact
Qnorm2_exact = inner(Q_exact, Q_exact)
lapQ_exact = div(grad(Q_exact))

bulk_exact = (
    -a2 * Q_exact
    - a3 * Q_exact_2
    + a4 * Qnorm2_exact * Q_exact
)

f_tensor = -l1 * lapQ_exact + (1/eta**2) * bulk_exact


# ----------------------------------------------------
# Manufactured surface forcing
#
# Natural boundary condition:
#
# l1 dQ/dnu
# + w0 Q
# + w1 (Q - Qperp)
# + (w2/omega)(|Qtilde|^2 - s0^2) Qtilde
# = G
#
# We choose G so that Q_exact satisfies the boundary condition.
# ----------------------------------------------------
nu = FacetNormal(mesh)

Pi = I - outer(nu, nu)

Qperp_exact = Pi * Q_exact * Pi
Qtilde_exact = Q_exact + (s0/2) * I

surface_cubic_exact = (
    (inner(Qtilde_exact, Qtilde_exact) - s0**2)
    * Qtilde_exact
)

G_tensor = (
    l1 * dot(grad(Q_exact), nu)
    + w0 * Q_exact
    + w1 * (Q_exact - Qperp_exact)
    + (w2/omega) * surface_cubic_exact
)


# ----------------------------------------------------
# Initial guess
# No DirichletBC.
# Boundary condition is natural Robin/MMS boundary forcing.
# ----------------------------------------------------
q.interpolate(as_vector([
    0.05*(x - 0.5),
    0.05*(y - 0.5)
]))


# ----------------------------------------------------
# Gradient descent parameters
# ----------------------------------------------------
tau = Constant(1.0e-4)

max_iter = 10000
tol = 1.0e-5


# ----------------------------------------------------
# Tensor versions
# ----------------------------------------------------
Q_trial = Q_tensor(u)
Q_old = Q_tensor(q)
P = Q_tensor(p)


# ----------------------------------------------------
# Explicit nonlinear bulk term evaluated at q^k
# ----------------------------------------------------
Q_old_2 = Q_old * Q_old
Qnorm2_old = inner(Q_old, Q_old)

bulk_old = (
    -a2 * Q_old
    - a3 * Q_old_2
    + a4 * Qnorm2_old * Q_old
)


# ----------------------------------------------------
# Boundary quantities for unknown and old iterate
# ----------------------------------------------------
Qperp_trial = Pi * Q_trial * Pi

Qtilde_old = Q_old + (s0/2) * I

surface_cubic_old = (
    (inner(Qtilde_old, Qtilde_old) - s0**2)
    * Qtilde_old
)


# ----------------------------------------------------
# Gradient descent weak form with manufactured surface term
#
# Steady problem:
#
# l1 (grad Q, grad P)
# + w0 (Q, P)_Gamma
# + w1 (Q - Qperp, P)_Gamma
# + 1/eta^2 (bulk(Q), P)
# + w2/omega (surface_cubic(Q), P)_Gamma
# =
# (f, P)_Omega + (G, P)_Gamma
#
# Gradient descent:
#
# (Q^{k+1} - Q^k)/tau + steady residual = 0
#
# We treat:
#   l1 term implicitly
#   w0 term implicitly
#   w1 term implicitly
#   bulk term explicitly
#   surface cubic term explicitly
# ----------------------------------------------------
A = (
    inner(Q_trial / tau, P) * dx
    + l1 * inner(grad(Q_trial), grad(P)) * dx
    + w0 * inner(Q_trial, P) * ds
    + w1 * inner(Q_trial - Qperp_trial, P) * ds
)

L = (
    inner(Q_old / tau, P) * dx
    - (1/eta**2) * inner(bulk_old, P) * dx
    + inner(f_tensor, P) * dx
    + inner(G_tensor, P) * ds
    - (w2/omega) * inner(surface_cubic_old, P) * ds
)


problem = LinearVariationalProblem(A, L, q_new)

solver = LinearVariationalSolver(
    problem,
    solver_parameters={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
)


# ----------------------------------------------------
# Gradient descent loop
# ----------------------------------------------------
for k in range(max_iter):

    solver.solve()

    update_norm = sqrt(assemble(inner(q_new - q, q_new - q) * dx))

    q.assign(q_new)

    if k % 100 == 0:
        Q_h = Q_tensor(q)
        error_L2 = sqrt(assemble(inner(Q_h - Q_exact, Q_h - Q_exact) * dx))

        print(
            f"iter = {k:5d}, "
            f"update = {update_norm:.3e}, "
            f"L2 error = {error_L2:.3e}"
        )

    if update_norm < tol:
        print(f"Converged at iter = {k}, update = {update_norm:.3e}")
        break


# ----------------------------------------------------
# Final tensor solution
# ----------------------------------------------------
Q_h = Q_tensor(q)


# ----------------------------------------------------
# L2 error
# ----------------------------------------------------
error_L2 = sqrt(assemble(inner(Q_h - Q_exact, Q_h - Q_exact) * dx))
print("Final L2 error in Q =", error_L2)


# ----------------------------------------------------
# Pointwise error
# ----------------------------------------------------
Sspace = FunctionSpace(mesh, "CG", 1)

pointwise_error = Function(Sspace, name="pointwise_error")

pointwise_error.project(
    sqrt(inner(Q_h - Q_exact, Q_h - Q_exact))
)

max_error = pointwise_error.dat.data_ro.max()

print("Max pointwise error in Q =", max_error)

VTKFile("pointwise_error_mms_surface_gradient_descent.pvd").write(pointwise_error)


# ----------------------------------------------------
# Save Q components
# ----------------------------------------------------
Q00 = Function(Sspace, name="Q_00")
Q01 = Function(Sspace, name="Q_01")
Q10 = Function(Sspace, name="Q_10")
Q11 = Function(Sspace, name="Q_11")

Q00.project(Q_h[0, 0])
Q01.project(Q_h[0, 1])
Q10.project(Q_h[1, 0])
Q11.project(Q_h[1, 1])

scalar_order = Function(Sspace, name="scalar_order")
scalar_order.project(sqrt(2 * inner(Q_h, Q_h)))


# ----------------------------------------------------
# Director field
# ----------------------------------------------------
W = VectorFunctionSpace(mesh, "CG", 1, dim=3)
director = Function(W, name="director")

q_vals = q.dat.data_ro
d_vals = director.dat.data

for i in range(len(q_vals)):

    q1 = q_vals[i, 0]
    q2 = q_vals[i, 1]

    Qmat = np.array([
        [q1,  q2],
        [q2, -q1]
    ])

    eigvals, eigvecs = np.linalg.eigh(Qmat)

    idx = np.argmax(eigvals)
    n = eigvecs[:, idx]

    d_vals[i, 0] = n[0]
    d_vals[i, 1] = n[1]
    d_vals[i, 2] = 0.0


# ----------------------------------------------------
# Save director line-field PNG
# ----------------------------------------------------
coords = mesh.coordinates.dat.data_ro
cells = mesh.coordinates.cell_node_map().values

triang = mtri.Triangulation(coords[:, 0], coords[:, 1], cells)

dvals = director.dat.data_ro

plt.figure(figsize=(6, 6))
plt.triplot(triang, linewidth=0.2, color="lightgray")

skip = 3
Lline = 0.012

for i in range(0, len(coords), skip):

    x0 = coords[i, 0]
    y0 = coords[i, 1]

    nx = dvals[i, 0]
    ny = dvals[i, 1]

    plt.plot(
        [x0 - Lline*nx, x0 + Lline*nx],
        [y0 - Lline*ny, y0 + Lline*ny],
        linewidth=0.8,
        color="black"
    )

plt.gca().set_aspect("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Director line field: MMS with surface terms")
plt.tight_layout()
plt.savefig("director_linefield_mms_surface_gradient_descent.png", dpi=300)
plt.close()

print("Saved director_linefield_mms_surface_gradient_descent.png")


# ----------------------------------------------------
# Save PVD files
# ----------------------------------------------------
VTKFile("q_solution_mms_surface_gradient_descent.pvd").write(q)
VTKFile("Q_components_mms_surface_gradient_descent.pvd").write(Q00, Q01, Q10, Q11)
VTKFile("director_mms_surface_gradient_descent.pvd").write(director)
VTKFile("scalar_order_mms_surface_gradient_descent.pvd").write(scalar_order)

print("Saved MMS surface gradient descent files.")