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
#
# So Q is automatically symmetric and traceless.
# ----------------------------------------------------
V = VectorFunctionSpace(mesh, "CG", 1, dim=2)

q = Function(V, name="q")          # current iterate q^k
q_new = Function(V, name="q_new")  # next iterate q^{k+1}

u = TrialFunction(V)               # unknown in each linear solve
p = TestFunction(V)                # test function

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


# ----------------------------------------------------
# Convert q-vector into Q-tensor
# ----------------------------------------------------
def Q_tensor(v):
    v1 = v[0]
    v2 = v[1]

    return as_matrix([
        [v1,  v2],
        [v2, -v1]
    ])


# ----------------------------------------------------
# Exact manufactured solution
#
# n = (x - 0.5, y - 0.5) / r
#
# Q_exact = s0 (n tensor n - I/2)
#
# We add eps to r^2 to avoid singularity at the center.
# ----------------------------------------------------
X = x - 0.5
Y = y - 0.5
r2 = X**2 + Y**2 + eps

Q_exact = s0 * (
    as_matrix([
        [X**2, X*Y],
        [X*Y, Y**2]
    ]) / r2
    - Identity(2)/2
)

q_exact = as_vector([
    Q_exact[0, 0],
    Q_exact[0, 1]
])


# ----------------------------------------------------
# Manufactured forcing
#
# Strong form:
#
# -l1 ΔQ + 1/eta^2 (-a2 Q - a3 Q^2 + a4 |Q|^2 Q) = f
#
# We plug Q_exact into the left side to define f.
# ----------------------------------------------------
Q_exact_2 = Q_exact * Q_exact
Qnorm2_exact = inner(Q_exact, Q_exact)
lapQ_exact = div(grad(Q_exact))

f_tensor = (
    -l1 * lapQ_exact
    + (1/eta**2) * (
        -a2 * Q_exact
        - a3 * Q_exact_2
        + a4 * Qnorm2_exact * Q_exact
    )
)


# ----------------------------------------------------
# Boundary condition
# ----------------------------------------------------
bc = DirichletBC(V, q_exact, "on_boundary")


# ----------------------------------------------------
# Initial guess
# ----------------------------------------------------
q.interpolate(as_vector([
    0.05*(x - 0.5),
    0.05*(y - 0.5)
]))

bc.apply(q)


# ----------------------------------------------------
# Gradient descent parameters
#
# tau is the artificial time step.
# If the method blows up, decrease tau.
# ----------------------------------------------------
tau = Constant(1.0e-5)

max_iter = 20000
tol = 1.0e-10


# ----------------------------------------------------
# Tensor versions of functions
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
# Gradient descent weak form
#
# Continuous gradient flow idea:
#
# Q_t = - gradient of energy
#
# Time discretization:
#
# (Q^{k+1} - Q^k)/tau
# - l1 ΔQ^{k+1}
# + nonlinear_bulk(Q^k)
# = f
#
# Weak form:
#
# ((Q^{k+1})/tau, P)
# + l1 (grad Q^{k+1}, grad P)
#
# =
#
# ((Q^k)/tau, P)
# - 1/eta^2 (bulk_old, P)
# + (f, P)
# ----------------------------------------------------
A = (
    inner(Q_trial / tau, P) * dx
    + l1 * inner(grad(Q_trial), grad(P)) * dx
)

L = (
    inner(Q_old / tau, P) * dx
    - (1/eta**2) * inner(bulk_old, P) * dx
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
# Gradient descent loop
# ----------------------------------------------------
for k in range(max_iter):

    solver.solve()

    update_norm = sqrt(assemble(inner(q_new - q, q_new - q) * dx))

    q.assign(q_new)
    bc.apply(q)

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

VTKFile("pointwise_error_gradient_descent.pvd").write(pointwise_error)


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
#
# At each node, compute the eigenvector of Q with
# largest eigenvalue.
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
plt.triplot(triang, linewidth=0.2)

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
        linewidth=0.8
    )

plt.gca().set_aspect("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Director line field from gradient descent")
plt.tight_layout()
plt.savefig("director_linefield_gradient_descent.png", dpi=300)
plt.close()

print("Saved director_linefield_gradient_descent.png")


# ----------------------------------------------------
# Save PVD files
# ----------------------------------------------------
VTKFile("q_solution_gradient_descent.pvd").write(q)
VTKFile("Q_components_gradient_descent.pvd").write(Q00, Q01, Q10, Q11)
VTKFile("director_gradient_descent.pvd").write(director)
VTKFile("scalar_order_gradient_descent.pvd").write(scalar_order)

print("Saved gradient descent files.")