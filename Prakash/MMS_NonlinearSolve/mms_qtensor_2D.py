import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from firedrake import *
from firedrake.output import VTKFile

import numpy as np

# ----------------------------------------------------
# Mesh: plain unit square
# ----------------------------------------------------
mesh = UnitSquareMesh(64, 64)

# ----------------------------------------------------
# Unknown space
# q = (q1, q2)
# Q = [[q1, q2],
#      [q2, -q1]]
# ----------------------------------------------------
V = VectorFunctionSpace(mesh, "CG", 1, dim=2)

q = Function(V, name="q")
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

# ----------------------------------------------------
# Helper: vector -> symmetric traceless tensor
# ----------------------------------------------------
def Q_tensor(v):
    v1, v2 = v[0], v[1]
    return as_matrix([
        [v1,  v2],
        [v2, -v1]
    ])

Q = Q_tensor(q)
P = Q_tensor(p)

# ----------------------------------------------------
# Regularized manufactured Q
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
# Forcing tensor from strong form
# ----------------------------------------------------
Q_exact_2 = Q_exact * Q_exact
Qnorm2_exact = inner(Q_exact, Q_exact)
lapQ_exact = div(grad(Q_exact))

f_tensor = (
    -l1 * lapQ_exact
    + (1/eta**2) * (
        -a2 * Q_exact
        -a3 * Q_exact_2
        + a4 * Qnorm2_exact * Q_exact
    )
)

# ----------------------------------------------------
# Weak form written with matrix tensors
# ----------------------------------------------------
Q2 = Q * Q
Qnorm2 = inner(Q, Q)

F = (
    l1 * inner(grad(Q), grad(P)) * dx
    + (1/eta**2) * inner(
        -a2*Q - a3*Q2 + a4*Qnorm2*Q,
        P
    ) * dx
    - inner(f_tensor, P) * dx
)

# ----------------------------------------------------
# Boundary condition on whole square boundary
# ----------------------------------------------------
bc = DirichletBC(V, q_exact, "on_boundary")

# ----------------------------------------------------
# Initial guess, not exact
# ----------------------------------------------------
q.assign(project(as_vector([
    0.05*(x - 0.5),
    0.05*(y - 0.5)
]), V))

# ----------------------------------------------------
# Solve nonlinear system
# ----------------------------------------------------
problem = NonlinearVariationalProblem(F, q, bcs=[bc])

solver = NonlinearVariationalSolver(
    problem,
    solver_parameters={
        "snes_type": "newtonls",
        "snes_monitor": None,
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
)

solver.solve()

# ----------------------------------------------------
# Error in full tensor Q
# ----------------------------------------------------
Q_h = Q_tensor(q)

error_L2 = sqrt(assemble(inner(Q_h - Q_exact, Q_h - Q_exact)*dx))
print("L2 error in Q =", error_L2)

# ----------------------------------------------------
# Pointwise error field
# ----------------------------------------------------
Sspace = FunctionSpace(mesh, "CG", 1)

pointwise_error = Function(Sspace, name="pointwise_error")

pointwise_error.assign(project(
    sqrt(inner(Q_h - Q_exact, Q_h - Q_exact)),
    Sspace
))

max_error = pointwise_error.dat.data_ro.max()
print("Max pointwise error in Q =", max_error)

VTKFile("pointwise_error.pvd").write(pointwise_error)

# ----------------------------------------------------
# Output fields
# ----------------------------------------------------
Sspace = FunctionSpace(mesh, "CG", 1)

Q00 = Function(Sspace, name="Q_00")
Q01 = Function(Sspace, name="Q_01")
Q10 = Function(Sspace, name="Q_10")
Q11 = Function(Sspace, name="Q_11")

Q00.assign(project(Q_h[0, 0], Sspace))
Q01.assign(project(Q_h[0, 1], Sspace))
Q10.assign(project(Q_h[1, 0], Sspace))
Q11.assign(project(Q_h[1, 1], Sspace))

scalar_order = Function(Sspace, name="scalar_order")
scalar_order.assign(project(sqrt(2*inner(Q_h, Q_h)), Sspace))

# ----------------------------------------------------
# Director field using eigenvector of largest eigenvalue
# ----------------------------------------------------
W = VectorFunctionSpace(mesh, "CG", 1, dim=3)
director = Function(W, name="director")

# Work with nodal values of q
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

    # largest eigenvalue
    idx = np.argmax(eigvals)
    n = eigvecs[:, idx]

    d_vals[i, 0] = n[0]
    d_vals[i, 1] = n[1]
    d_vals[i, 2] = 0.0

# ----------------------------------------------------
# Save director PNG in current MMS folder
# Note that plt.plot(...) automatically cycles through Matplotlib’s default color list every time you draw a new line,
# so the lines will have different colors.
# ----------------------------------------------------
coords = mesh.coordinates.dat.data_ro
cells = mesh.coordinates.cell_node_map().values

triang = mtri.Triangulation(coords[:, 0], coords[:, 1], cells)
dvals = director.dat.data_ro

plt.figure(figsize=(6, 6))
plt.triplot(triang, linewidth=0.2)

skip = 3
L = 0.012

for i in range(0, len(coords), skip):
    x0 = coords[i, 0]
    y0 = coords[i, 1]

    nx = dvals[i, 0]
    ny = dvals[i, 1]

    plt.plot(
        [x0 - L*nx, x0 + L*nx],
        [y0 - L*ny, y0 + L*ny],
        linewidth=0.8
    )

plt.gca().set_aspect("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Director line field")
plt.tight_layout()
plt.savefig("director_linefield.png", dpi=300)
plt.close()

print("Saved director_linefield.png")

# ----------------------------------------------------
# Save PVD/VTU files in current MMS folder
# ----------------------------------------------------
VTKFile("q_solution.pvd").write(q)
VTKFile("Q_components.pvd").write(Q00, Q01, Q10, Q11)
VTKFile("director.pvd").write(director)
VTKFile("scalar_order.pvd").write(scalar_order)

print("Saved files in current MMS directory")