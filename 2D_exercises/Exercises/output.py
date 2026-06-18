import csv

from firedrake.output import VTKFile

from . import config


def make_vtk_file(mode_name, anchoring, N, exact_case_name=None, save_vtk=True):
    if not save_vtk:
        return None, None

    config.VTK_DIR.mkdir(parents=True, exist_ok=True)
    if exact_case_name is None:
        vtk_name = f"{mode_name}_{anchoring}_N{N}.pvd"
    else:
        vtk_name = f"{mode_name}_{exact_case_name}_{anchoring}_N{N}.pvd"
    vtk_path = config.VTK_DIR / vtk_name
    return VTKFile(str(vtk_path)), vtk_path


def write_vtk_frame(vtk_file, q_func, q_exact_fn, q_error, time_value=0.0):
    if vtk_file is None:
        return

    q_error.interpolate(q_func - q_exact_fn)
    vtk_file.write(q_func, q_exact_fn, q_error, time=time_value)


def write_direct_errors(rows):
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.OUT_DIR / "direct_errors.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "exact_case",
            "anchoring",
            "N",
            "h",
            "L2",
            "H1",
            "vtk_path",
        ])

        for row in rows:
            writer.writerow([
                row["exact_case"],
                row["anchoring"],
                row["N"],
                row["h"],
                row["L2"],
                row["H1"],
                row["vtk_path"] or "",
            ])

    return csv_path


def write_gradient_comparison(rows):
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.OUT_DIR / "gradient_comparison.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "exact_case",
            "anchoring",
            "N",
            "h",
            "L2",
            "H1",
            "final_energy",
            "final_update",
            "vtk_path",
        ])

        for row in rows:
            energy_history = row["energy_history"]
            update_history = row["update_history"]
            writer.writerow([
                row["exact_case"],
                row["anchoring"],
                row["N"],
                row["h"],
                row["L2"],
                row["H1"],
                energy_history[-1] if energy_history else "",
                update_history[-1] if update_history else "",
                row["vtk_path"] or "",
            ])

    return csv_path


def write_energy_history(exact_case_name, anchoring, energy_history):
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    energy_path = (
        config.OUT_DIR / f"energy_history_{exact_case_name}_{anchoring}.csv"
    )

    with open(energy_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "energy"])

        for k, energy in enumerate(energy_history):
            writer.writerow([k, energy])

    return energy_path
