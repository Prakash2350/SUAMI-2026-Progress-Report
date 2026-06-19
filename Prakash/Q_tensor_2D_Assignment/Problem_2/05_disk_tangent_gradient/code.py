import matplotlib
matplotlib.use("Agg")

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from firedrake import *
from firedrake.output import VTKFile


# ----------------------------------------------------
# Choices
# ----------------------------------------------------
mesh_sizes = [4]

#anchoring_type = "planar"
anchoring_type = "homeotropic"

OUTDIR = Path(__file__).resolve().parent


# ----------------------------------------------------
# Parameters from professor
# ----------------------------------------------------
s0 = Constant(0.7)

l1 = Constant(1.0)
eta = Constant(1)

a2 = Constant(7.5)
a3 = Constant(60.98)
a4 = Constant(66.52)

omega = Constant(0.1)

eps_a = Constant(0.0)

tau = Constant(1.0e-3)
max_iter = 2000
tol = 1.0e-5

if anchoring_type == "planar":
    w0 = Constant(0.0)
    w1 = Constant(10.0)
    w2 = Constant(10.0)

elif anchoring_type == "homeotropic":
    w0 = Constant(10.0)
    w1 = Constant(0.0)
    w2 = Constant(0.0)


# ----------------------------------------------------
# q = (q1,q2) to 2D traceless Q tensor
# ----------------------------------------------------
def Q_tensor(v):
    return as_matrix([
        [v[0],  v[1]],
        [v[1], -v[0]]
    ])


# ----------------------------------------------------
# Project Q components for saving
# ----------------------------------------------------
def project_Q_components(q, S):
    Q = Q_tensor(q)

    Q00 = Function(S, name="Q_00")
    Q01 = Function(S, name="Q_01")
    Q10 = Function(S, name="Q_10")
    Q11 = Function(S, name="Q_11")

    Q00.project(Q[0, 0])
    Q01.project(Q[0, 1])
    Q10.project(Q[1, 0])
    Q11.project(Q[1, 1])

    return Q00, Q01, Q10, Q11


# ----------------------------------------------------
# Compute director from Q
# ----------------------------------------------------
def compute_director(mesh, q):
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

    return director


# ----------------------------------------------------
# One mesh run
# ----------------------------------------------------
def run_one_mesh(N):

    print(f"\nRunning N = {N}, anchoring = {anchoring_type}")

    folder = OUTDIR / f"experiment_1_{anchoring_type}_N{N}"
    folder.mkdir(parents=True, exist_ok=True)

    mesh = UnitDiskMesh(N, N)

    V = VectorFunctionSpace(mesh, "CG", 1, dim=2)
    S = FunctionSpace(mesh, "CG", 1)

    q = Function(V, name="q")
    q_new = Function(V, name="q_new")

    u = TrialFunction(V)
    p = TestFunction(V)

    I = Identity(2)
    nu = FacetNormal(mesh)
    Pi = I - outer(nu, nu)
    x, y = SpatialCoordinate(mesh)

    # ------------------------------------------------
    # External electric field
    # Currently E = 0, but you can change this later.
    # ------------------------------------------------
    E = as_vector([1, 2])
    E_tensor = outer(E, E)
    forcing = -0.5 * eps_a * E_tensor

    # ------------------------------------------------
    # Initial director field n0
    # Problem 1: n0 = (1, 0)
    # ------------------------------------------------
    eps = 1.0e-3
    n0 = as_vector([y, -x])

    n0_length = sqrt(n0[0]**2 + n0[1]**2) + eps

    nx = n0[0] / n0_length
    ny = n0[1] / n0_length

    Q0 = s0 * (
        as_matrix([
            [nx*nx, nx*ny],
            [nx*ny, ny*ny]
        ])
        - I/2
    )

    q0 = as_vector([
        Q0[0, 0],
        Q0[0, 1]
    ])

    q.interpolate(q0)
    

    # ------------------------------------------------
    # Boundary target for homeotropic anchoring
    # ------------------------------------------------
    Q_Gamma = s0 * (outer(nu, nu) - I/2)

    # ------------------------------------------------
    # Semi-implicit gradient descent weak form
    # ------------------------------------------------
    Q_trial = Q_tensor(u)
    Q_old = Q_tensor(q)
    P = Q_tensor(p)

    Q_old_2 = Q_old * Q_old
    Qnorm2_old = inner(Q_old, Q_old)

    bulk_old = (
        -a2 * Q_old
        - a3 * Q_old_2
        + a4 * Qnorm2_old * Q_old
    )

    Qperp_trial = Pi * Q_trial * Pi

    Qtilde_old = Q_old + (s0/2) * I

    surface_cubic_old = (
        (inner(Qtilde_old, Qtilde_old) - s0**2)
        * Qtilde_old
    )

    A = (
        inner(Q_trial / tau, P) * dx
        + l1 * inner(grad(Q_trial), grad(P)) * dx
        + w0 * inner(Q_trial, P) * ds
        + w1 * inner(Q_trial - Qperp_trial, P) * ds
    )

    L = (
        inner(Q_old / tau, P) * dx
        - (1/eta**2) * inner(bulk_old, P) * dx
        + inner(forcing, P) * dx
        + w0 * inner(Q_Gamma, P) * ds
        - w1 * inner((s0/2) * outer(nu, nu), P) * ds
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

    # ------------------------------------------------
    # Energy
    # ------------------------------------------------
    def compute_energy():

        Q_h = Q_tensor(q)
        Q_h_2 = Q_h * Q_h
        Qnorm2 = inner(Q_h, Q_h)

        bulk_energy = (
            -a2/2 * Qnorm2
            - a3/3 * tr(Q_h_2 * Q_h)
            + a4/4 * Qnorm2**2
        )

        Qperp = Pi * Q_h * Pi
        Qtilde = Q_h + (s0/2) * I

        planar_term = Q_h - Qperp - (s0/2) * outer(nu, nu)

        energy = assemble(
            l1/2 * inner(grad(Q_h), grad(Q_h)) * dx
            + (1/eta**2) * bulk_energy * dx
            + w0/2 * inner(Q_h - Q_Gamma, Q_h - Q_Gamma) * ds
            + w1/2 * inner(planar_term, planar_term) * ds
            + (w2/(4*omega)) * (inner(Qtilde, Qtilde) - s0**2)**2 * ds
            - inner(forcing, Q_h) * dx
        )

        return float(energy)

    # ------------------------------------------------
    # Output files
    # ------------------------------------------------
    Q_file = VTKFile(str(folder / "Q_tensor_steps.pvd"))
    director_file = VTKFile(str(folder / "director_steps.pvd"))

    energies = []

    energy = compute_energy()
    energies.append([0, energy])

    Q00, Q01, Q10, Q11 = project_Q_components(q, S)
    director = compute_director(mesh, q)

    Q_file.write(Q00, Q01, Q10, Q11, time=0)
    director_file.write(director, time=0)

    # ------------------------------------------------
    # Gradient descent loop
    # ------------------------------------------------
    for k in range(max_iter):

        solver.solve()

        update_norm = sqrt(assemble(inner(q_new - q, q_new - q) * dx))
        q.assign(q_new)

        energy = compute_energy()
        energies.append([k+1, energy])

        if (k + 1) % 5 == 0: 
            Q00, Q01, Q10, Q11 = project_Q_components(q, S)
            director = compute_director(mesh, q)

            Q_file.write(Q00, Q01, Q10, Q11, time=k+1)
            director_file.write(director, time=k+1)

        if k % 50 == 0:
            print(
                f"iter = {k:5d}, "
                f"update = {update_norm:.3e}, "
                f"energy = {energy:.10e}"
            )

        if update_norm < tol:
            print(f"Converged at iter = {k}, update = {update_norm:.3e}")
            break

    # ------------------------------------------------
    # Save energy CSV and plot
    # ------------------------------------------------
    energy_csv = folder / "energy_history.csv"

    with open(energy_csv, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["iteration", "energy"])
        writer.writerows(energies)

    its = [row[0] for row in energies]
    Es = [row[1] for row in energies]

    plt.figure()
    plt.plot(its, Es)
    plt.xlabel("iteration")
    plt.ylabel("energy")
    plt.title(f"Energy vs iteration, N={N}, {anchoring_type}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(folder / "energy_vs_iteration.png", dpi=300)
    plt.close()

    return 1.0 / N, energies[-1][1]


# ----------------------------------------------------
# Run all meshes
# ----------------------------------------------------
summary = []

for N in mesh_sizes:
    h, final_energy = run_one_mesh(N)
    summary.append([N, h, final_energy])


# ----------------------------------------------------
# Save summary CSV
# ----------------------------------------------------
summary_csv = OUTDIR / f"experiment_1_summary_{anchoring_type}.csv"

with open(summary_csv, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["N", "h", "final_energy"])
    writer.writerows(summary)

print("Saved", summary_csv)
print("Done.")