import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from firedrake import *
from firedrake.output import VTKFile

CODE_DIR = Path(__file__).resolve().parent
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

gradient_N = 64
gradient_time_step_value = 1.0e-5
gradient_max_iter = 2000
gradient_tol = 1.0e-10
gradient_save_every = 50


def solve_one_mesh(N, anchoring, mode="direct", save_vtk=True):
    if anchoring not in {"homeotropic", "planar"}:
        raise ValueError("anchoring must be 'homeotropic' or 'planar'")

    mesh = UnitSquareMesh(N, N)

    V = VectorFunctionSpace(mesh, "CG", 1, dim=2)
    q = Function(V, name="q")
    p = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    normal = FacetNormal(mesh)
    I2 = Identity(2)

    s0 = Constant(s0_value)

    l1 = Constant(l1_value)
    eta = Constant(eta_value)

    a2 = Constant(a2_value)
    a3 = Constant(a3_value)
    a4 = Constant(a4_value)

    eps = Constant(eps_value)
    omega = Constant(omega_value)

    if anchoring == "homeotropic":
        w0 = Constant(homeotropic_w0)
        w1 = Constant(0.0)
        w2 = Constant(0.0)
    else:
        w0 = Constant(0.0)
        w1 = Constant(planar_w1)
        w2 = Constant(planar_w2)

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

    X = x - 0.5
    Y = y - 0.5
    r2 = X**2 + Y**2 + eps

    q_exact = as_vector([
        (s0 / 2.0) * (X**2 - Y**2) / r2,
        s0 * X * Y / r2,
    ])

    Q_exact = Q_tensor(q_exact)

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

    def make_vtk_file(mode_name):
        if not save_vtk:
            return None, None

        VTK_DIR.mkdir(parents=True, exist_ok=True)
        vtk_path = VTK_DIR / f"{mode_name}_{anchoring}_N{N}.pvd"
        return VTKFile(str(vtk_path)), vtk_path

    def write_vtk_frame(vtk_file, q_func, time_value=0.0):
        if vtk_file is None:
            return

        q_error.interpolate(q_func - q_exact_fn)
        vtk_file.write(q_func, q_exact_fn, q_error, time=time_value)

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

    if mode == "direct":
        Q = Q_tensor(q)
        P = Q_tensor(p)

        bulk_Q = (
            -a2 * Q
            - a3 * dot(Q, Q)
            + a4 * inner(Q, Q) * Q
        )

        F_direct = (
            l1 * inner(grad(Q), grad(P)) * dx
            + (1.0 / eta**2) * inner(bulk_Q, P) * dx
            - inner(f_tensor, P) * dx
            + inner(surface_grad(Q) - boundary_rhs, P) * ds
        )

        # Start near the manufactured solution so Newton selects the MMS branch.
        q.interpolate(q_exact)

        solve(
            F_direct == 0,
            q,
            solver_parameters={
                "snes_type": "newtonls",
                "snes_rtol": 1.0e-11,
                "snes_atol": 1.0e-12,
                "snes_max_it": 50,
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        )

        vtk_file, vtk_path = make_vtk_file("direct")
        write_vtk_frame(vtk_file, q)

        L2_err, H1_err = compute_errors(q)
        h = 1.0 / N

        return {
            "N": N,
            "h": h,
            "anchoring": anchoring,
            "L2": L2_err,
            "H1": H1_err,
            "energy_history": [],
            "update_history": [],
            "vtk_path": vtk_path,
        }

    if mode == "gradient":
        time_step = Constant(gradient_time_step_value)

        q_old = Function(V, name="q")
        q_new = Function(V, name="q_new")

        q_old.assign(0.0)
        q_new.assign(q_old)

        Q_old = Q_tensor(q_old)
        Q_new = Q_tensor(q_new)
        P = Q_tensor(p)

        bulk_new = (
            -a2 * Q_new
            - a3 * dot(Q_new, Q_new)
            + a4 * inner(Q_new, Q_new) * Q_new
        )

        F_gd = (
            inner((Q_new - Q_old) / time_step, P) * dx
            + l1 * inner(grad(Q_new), grad(P)) * dx
            + (1.0 / eta**2) * inner(bulk_new, P) * dx
            - inner(f_tensor, P) * dx
            + inner(surface_grad(Q_new) - boundary_rhs, P) * ds
        )

        vtk_file, vtk_path = make_vtk_file("gradient")
        write_vtk_frame(vtk_file, q_old, time_value=0.0)

        energy_history = []
        update_history = []

        E0 = float(assemble(energy_form(Q_old)))
        energy_history.append(E0)

        for k in range(1, gradient_max_iter + 1):
            q_new.assign(q_old)

            solve(
                F_gd == 0,
                q_new,
                solver_parameters={
                    "snes_type": "newtonls",
                    "snes_rtol": 1.0e-10,
                    "snes_atol": 1.0e-12,
                    "snes_max_it": 30,
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps",
                },
            )

            update_norm = math.sqrt(float(assemble(
                inner(Q_tensor(q_new) - Q_tensor(q_old),
                      Q_tensor(q_new) - Q_tensor(q_old)) * dx
            )))

            q_old.assign(q_new)

            energy_value = float(assemble(energy_form(Q_tensor(q_old))))
            energy_history.append(energy_value)
            update_history.append(update_norm)

            if k % gradient_save_every == 0:
                write_vtk_frame(
                    vtk_file,
                    q_old,
                    time_value=k * gradient_time_step_value,
                )

            if k % 50 == 0:
                L2_err, H1_err = compute_errors(q_old)
                print(
                    f"[{anchoring}] iter={k:5d}, "
                    f"energy={energy_value:.12e}, "
                    f"update={update_norm:.3e}, "
                    f"L2={L2_err:.3e}, "
                    f"H1={H1_err:.3e}"
                )

            if update_norm < gradient_tol:
                print(
                    f"[{anchoring}] converged at iter={k}, "
                    f"update={update_norm:.3e}"
                )
                write_vtk_frame(
                    vtk_file,
                    q_old,
                    time_value=k * gradient_time_step_value,
                )
                break

        q.assign(q_old)

        L2_err, H1_err = compute_errors(q)
        h = 1.0 / N

        return {
            "N": N,
            "h": h,
            "anchoring": anchoring,
            "L2": L2_err,
            "H1": H1_err,
            "energy_history": energy_history,
            "update_history": update_history,
            "vtk_path": vtk_path,
        }

    raise ValueError("mode must be 'direct' or 'gradient'")


def run_direct_convergence(save_vtk=True):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for anchoring in ["homeotropic", "planar"]:
        print(f"\nDirect solve convergence: {anchoring}")

        for N in mesh_sizes:
            result = solve_one_mesh(
                N,
                anchoring,
                mode="direct",
                save_vtk=save_vtk,
            )
            rows.append(result)

            print(
                f"N={N:4d}, "
                f"h={result['h']:.6e}, "
                f"L2={result['L2']:.6e}, "
                f"H1={result['H1']:.6e}"
            )

    csv_path = OUT_DIR / "direct_errors.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["anchoring", "N", "h", "L2", "H1", "vtk_path"])

        for r in rows:
            writer.writerow([
                r["anchoring"],
                r["N"],
                r["h"],
                r["L2"],
                r["H1"],
                r["vtk_path"] or "",
            ])

    return rows


def plot_direct_errors(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for anchoring in ["homeotropic", "planar"]:
        data = [r for r in rows if r["anchoring"] == anchoring]

        hs = [r["h"] for r in data]
        L2s = [r["L2"] for r in data]
        H1s = [r["H1"] for r in data]

        plt.figure()
        plt.loglog(hs, L2s, "o-", label="L2 error")
        plt.loglog(hs, H1s, "s-", label="H1 error")
        plt.gca().invert_xaxis()
        plt.xlabel("h")
        plt.ylabel("error")
        plt.title(f"Direct solve MMS errors: {anchoring}")
        plt.legend()
        plt.grid(True, which="both")
        plt.savefig(OUT_DIR / f"direct_errors_{anchoring}.png", dpi=200)
        plt.close()


def run_gradient_flows(save_vtk=True):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for anchoring in ["homeotropic", "planar"]:
        print(f"\nGradient flow: {anchoring}")

        result = solve_one_mesh(
            gradient_N,
            anchoring,
            mode="gradient",
            save_vtk=save_vtk,
        )
        results.append(result)

        energy_path = OUT_DIR / f"energy_history_{anchoring}.csv"

        with open(energy_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "energy"])

            for k, E in enumerate(result["energy_history"]):
                writer.writerow([k, E])

        plt.figure()
        plt.plot(range(len(result["energy_history"])), result["energy_history"])
        plt.xlabel("iteration")
        plt.ylabel("energy")
        plt.title(f"Gradient flow energy: {anchoring}")
        plt.grid(True)
        plt.savefig(OUT_DIR / f"energy_{anchoring}.png", dpi=200)
        plt.close()

        energies = result["energy_history"]
        max_increase = max(
            (
                energies[k + 1] - energies[k]
                for k in range(len(energies) - 1)
            ),
            default=0.0,
        )

        print(
            f"[{anchoring}] final L2={result['L2']:.6e}, "
            f"final H1={result['H1']:.6e}, "
            f"max energy increase={max_increase:.6e}"
        )

    return results


if __name__ == "__main__":
    direct_rows = run_direct_convergence(save_vtk=True)
    plot_direct_errors(direct_rows)
    run_gradient_flows(save_vtk=True)

    print(
        "\nSaved results in:\n"
        f"  {OUT_DIR}\n\n"
        "VTK files in:\n"
        f"  {VTK_DIR}"
    )
