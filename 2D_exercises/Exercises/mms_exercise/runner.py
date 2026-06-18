import time

from . import config
from .output import (
    write_actual_comparison,
    write_direct_errors,
    write_energy_history,
    write_gradient_comparison,
)
from .plots import plot_direct_errors, plot_energy_history
from .solvers import solve_one_mesh


def run_direct_convergence(save_vtk=True):
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for anchoring in ["homeotropic", "planar"]:
        print(f"\nDirect solve convergence: {anchoring}")

        for N in config.mesh_sizes:
            for exact_case_name in config.Q_EXACT_CASES:
                result = solve_one_mesh(
                    N,
                    anchoring,
                    mode="direct",
                    save_vtk=save_vtk,
                    exact_case_name=exact_case_name,
                    problem_mode="mms",
                )
                rows.append(result)

                print(
                    f"case={exact_case_name}, "
                    f"N={N:4d}, "
                    f"h={result['h']:.6e}, "
                    f"L2={result['L2']:.6e}, "
                    f"H1={result['H1']:.6e}"
                )

    write_direct_errors(rows)
    return rows


def run_gradient_flows(save_vtk=True):
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for exact_case_name in config.Q_EXACT_CASES:
        for anchoring in ["homeotropic", "planar"]:
            print(f"\nGradient flow: {exact_case_name}, {anchoring}")

            result = solve_one_mesh(
                config.gradient_N,
                anchoring,
                mode="gradient",
                save_vtk=save_vtk,
                exact_case_name=exact_case_name,
                problem_mode="mms",
            )
            results.append(result)

            write_energy_history(
                exact_case_name,
                anchoring,
                result["energy_history"],
            )
            plot_energy_history(
                exact_case_name,
                anchoring,
                result["energy_history"],
            )

            energies = result["energy_history"]
            max_increase = max(
                (
                    energies[k + 1] - energies[k]
                    for k in range(len(energies) - 1)
                ),
                default=0.0,
            )

            print(
                f"[{exact_case_name}, {anchoring}] "
                f"final L2={result['L2']:.6e}, "
                f"final H1={result['H1']:.6e}, "
                f"max energy increase={max_increase:.6e}"
            )

    write_gradient_comparison(results)
    return results


def run_actual_gradient_flows(save_vtk=True):
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for anchoring in ["homeotropic", "planar"]:
        print(f"\nActual gradient flow: {anchoring}")
        start_time = time.perf_counter()

        result = solve_one_mesh(
            config.gradient_N,
            anchoring,
            mode="gradient",
            save_vtk=save_vtk,
            problem_mode="actual",
        )
        results.append(result)

        write_energy_history(
            result["exact_case"],
            anchoring,
            result["energy_history"],
        )
        plot_energy_history(
            result["exact_case"],
            anchoring,
            result["energy_history"],
        )

        energies = result["energy_history"]
        updates = result["update_history"]
        elapsed = time.perf_counter() - start_time
        print(
            f"[actual, {anchoring}] "
            f"final energy={energies[-1]:.12e}, "
            f"final update={updates[-1] if updates else 0.0:.3e}, "
            f"elapsed={elapsed:.2f}s"
        )

    write_actual_comparison(results)
    return results


def main():
    if config.problem_mode == "actual":
        run_actual_gradient_flows(save_vtk=True)
    elif config.problem_mode == "mms":
        direct_rows = run_direct_convergence(save_vtk=False)
        plot_direct_errors(direct_rows)
        run_gradient_flows(save_vtk=True)
    else:
        raise ValueError("problem_mode must be 'actual' or 'mms'")

    print(
        "\nSaved results in:\n"
        f"  {config.OUT_DIR}\n\n"
        "Each Q_exact case has its own folder under this directory."
    )
