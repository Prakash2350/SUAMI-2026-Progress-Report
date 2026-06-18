import matplotlib
matplotlib.use("Agg")

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from firedrake import *
from firedrake.output import VTKFile


# ----------------------------------------------------
# Choices
# ----------------------------------------------------
mesh_sizes = [16, 32, 64, 128]

# Choose one
anchoring_type = "planar"
# anchoring_type = "homeotropic"

OUTDIR = Path(__file__).resolve().parent


# ----------------------------------------------------
# Parameters
# ----------------------------------------------------
s0 = Constant(0.7)

l1 = Constant(1.0)
eta = Constant(0.1)

a2 = Constant(7.5)
a3 = Constant(61.0)
a4 = Constant(66.52)

omega = Constant(0.1)

if anchoring_type == "planar":
    w0 = Constant(0.0)
    w1 = Constant(10.0)
    w2 = Constant(10.0)

elif anchoring_type == "homeotropic":
    w0 = Constant(10.0)
    w1 = Constant(0.0)
    w2 = Constant(0.0)


# ----------------------------------------------------
# q vector to Q tensor
# ----------------------------------------------------
def Q_tensor(v):
    return as_matrix([
        [v[0],  v[1]],
        [v[1], -v[0]]
    ])


# ----------------------------------------------------
# Save director PNG
# ----------------------------------------------------
def save_director_png(mesh, q, filename):

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
        n = eigvecs[:, np.argmax(eigvals)]

        d_vals[i, 0] = n[0]
        d_vals[i, 1] = n[1]
        d_vals[i, 2] = 0.0

    coords = mesh.coordinates.dat.data_ro
    cells = mesh.coordinates.cell_node_map().values

    triang = mtri.Triangulation(coords[:, 0], coords[:, 1], cells)

    plt.figure(figsize=(6, 6))
    plt.triplot(triang, linewidth=0.2, color="lightgray")

    skip = max(1, len(coords)//700)
    Lline = 0.35 / np.sqrt(len(coords))

    for i in range(0, len(coords), skip):

        x0 = coords[i, 0]
        y0 = coords[i, 1]

        nx = d_vals[i, 0]
        ny = d_vals[i, 1]

        plt.plot(
            [x0 - Lline*nx, x0 + Lline*nx],
            [y0 - Lline*ny, y0 + Lline*ny],
            linewidth=0.8,
            color="black"
        )

    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Director line field")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# ----------------------------------------------------
# One solve
# ----------------------------------------------------
def run_one_mesh(N):

    print(f"\nRunning N = {N}, anchoring = {anchoring_type}")

    folder = OUTDIR / f"{anchoring_type}_N{N}"
    folder.mkdir(parents=True, exist_ok=True)

    mesh = UnitSquareMesh(N, N)

    V = VectorFunctionSpace(mesh, "CG", 1, dim=2)
    S = FunctionSpace(mesh, "CG", 1)

    q = Function(V, name="q")
    p = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    I = Identity(2)
    nu = FacetNormal(mesh)
    Pi = I - outer(nu, nu)

    # ------------------------------------------------
    # Type your director field here.
    #
    # For this file: n = (1, 0)
    # ------------------------------------------------
    eps = Constant(0.0)

    #nx_raw = 1.0 + 0*x
    #ny_raw = 0.0 + 0*x

    # Examples for later:
    #nx_raw = cos(pi*x)
    #ny_raw = sin(pi*x)

    nx_raw = x - 0.5
    ny_raw = y - 0.5
    eps = Constant(1.0e-5)

    length = sqrt(nx_raw**2 + ny_raw**2 + eps)

    nx = nx_raw / length
    ny = ny_raw / length

    # ------------------------------------------------
    # Automatically compute Q_exact from n
    # Q = s0(n tensor n - I/2)
    # ------------------------------------------------
    Q_exact = s0 * (
        as_matrix([
            [nx*nx, nx*ny],
            [nx*ny, ny*ny]
        ])
        - I/2
    )

    q_exact = as_vector([
        Q_exact[0, 0],
        Q_exact[0, 1]
    ])

    Q = Q_tensor(q)
    P = Q_tensor(p)

    # ------------------------------------------------
    # Manufactured volume force
    # -l1 Delta Q + bulk(Q) = f
    # ------------------------------------------------
    Q_exact_2 = Q_exact * Q_exact
    Qnorm2_exact = inner(Q_exact, Q_exact)
    lapQ_exact = div(grad(Q_exact))

    bulk_exact = (
        -a2 * Q_exact
        - a3 * Q_exact_2
        + a4 * Qnorm2_exact * Q_exact
    )

    f_tensor = -l1 * lapQ_exact + (1/eta**2) * bulk_exact

    # ------------------------------------------------
    # Manufactured surface force
    # ------------------------------------------------
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

    # ------------------------------------------------
    # Direct nonlinear weak form
    # ------------------------------------------------
    Q2 = Q * Q
    Qnorm2 = inner(Q, Q)

    bulk = (
        -a2 * Q
        - a3 * Q2
        + a4 * Qnorm2 * Q
    )

    Qperp = Pi * Q * Pi
    Qtilde = Q + (s0/2) * I

    surface_cubic = (
        (inner(Qtilde, Qtilde) - s0**2)
        * Qtilde
    )

    F = (
        l1 * inner(grad(Q), grad(P)) * dx
        + (1/eta**2) * inner(bulk, P) * dx
        + w0 * inner(Q, P) * ds
        + w1 * inner(Q - Qperp, P) * ds
        + (w2/omega) * inner(surface_cubic, P) * ds
        - inner(f_tensor, P) * dx
        - inner(G_tensor, P) * ds
    )

    q.interpolate(q_exact)

    problem = NonlinearVariationalProblem(F, q)

    solver = NonlinearVariationalSolver(
        problem,
        solver_parameters={
            "snes_type": "newtonls",
            "snes_rtol": 1.0e-10,
            "snes_atol": 1.0e-12,
            "snes_max_it": 50,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )

    solver.solve()

    Q_h = Q_tensor(q)

    # ------------------------------------------------
    # Errors
    # ------------------------------------------------
    error_L2 = sqrt(assemble(inner(Q_h - Q_exact, Q_h - Q_exact) * dx))

    error_H1_semi = sqrt(
        assemble(inner(grad(Q_h - Q_exact), grad(Q_h - Q_exact)) * dx)
    )

    error_H1 = sqrt(error_L2**2 + error_H1_semi**2)

    print("L2 error =", error_L2)
    print("H1 error =", error_H1)

    # ------------------------------------------------
    # Save Q tensor components only
    # ------------------------------------------------
    Q00 = Function(S, name="Q_00")
    Q01 = Function(S, name="Q_01")
    Q10 = Function(S, name="Q_10")
    Q11 = Function(S, name="Q_11")

    Q00.project(Q_h[0, 0])
    Q01.project(Q_h[0, 1])
    Q10.project(Q_h[1, 0])
    Q11.project(Q_h[1, 1])

    VTKFile(str(folder / "Q_tensor.pvd")).write(Q00, Q01, Q10, Q11)

    # ------------------------------------------------
    # Save director PNG only
    # ------------------------------------------------
    save_director_png(mesh, q, folder / "director.png")

    h = 1.0 / N

    return h, float(error_L2), float(error_H1)


# ----------------------------------------------------
# Run all meshes
# ----------------------------------------------------
results = []

for N in mesh_sizes:
    h, L2_error, H1_error = run_one_mesh(N)
    results.append([N, h, L2_error, H1_error])


# ----------------------------------------------------
# Save CSV
# ----------------------------------------------------
csv_name = OUTDIR / f"convergence_errors_{anchoring_type}.csv"

with open(csv_name, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["N", "h", "L2_error", "H1_error"])

    for row in results:
        writer.writerow(row)

print("Saved", csv_name)


# ----------------------------------------------------
# Plot errors
# ----------------------------------------------------
hs = [row[1] for row in results]
L2s = [row[2] for row in results]
H1s = [row[3] for row in results]

plt.figure()
plt.loglog(hs, L2s, "o-")
plt.gca().invert_xaxis()
plt.xlabel("h")
plt.ylabel("L2 error")
plt.title("L2 error vs h")
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig(OUTDIR / f"L2_error_vs_h_{anchoring_type}.png", dpi=300)
plt.close()

plt.figure()
plt.loglog(hs, H1s, "o-")
plt.gca().invert_xaxis()
plt.xlabel("h")
plt.ylabel("H1 error")
plt.title("H1 error vs h")
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig(OUTDIR / f"H1_error_vs_h_{anchoring_type}.png", dpi=300)
plt.close()

print("Done.")