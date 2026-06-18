import csv

from firedrake.output import VTKFile

from . import config


def case_output_dir(exact_case_name):
    return config.OUT_DIR / exact_case_name


def case_vtk_dir(exact_case_name):
    return case_output_dir(exact_case_name) / "vtk"


def rows_by_exact_case(rows):
    return [
        (
            exact_case_name,
            [row for row in rows if row["exact_case"] == exact_case_name],
        )
        for exact_case_name in sorted({row["exact_case"] for row in rows})
    ]


def make_vtk_file(mode_name, anchoring, N, exact_case_name=None, save_vtk=True):
    if not save_vtk:
        return None, None

    if exact_case_name is None:
        config.VTK_DIR.mkdir(parents=True, exist_ok=True)
        vtk_name = f"{mode_name}_{anchoring}_N{N}.pvd"
        vtk_path = config.VTK_DIR / vtk_name
    else:
        vtk_dir = case_vtk_dir(exact_case_name)
        vtk_dir.mkdir(parents=True, exist_ok=True)
        vtk_name = f"{mode_name}_{anchoring}_N{N}.pvd"
        vtk_path = vtk_dir / vtk_name

    return VTKFile(str(vtk_path)), vtk_path


def write_vtk_frame(vtk_file, q_func, q_exact_fn=None, q_error=None, time_value=0.0):
    if vtk_file is None:
        return

    if q_exact_fn is None or q_error is None:
        vtk_file.write(q_func, time=time_value)
        return

    q_error.interpolate(q_func - q_exact_fn)
    vtk_file.write(q_func, q_exact_fn, q_error, time=time_value)


def write_direct_errors(rows):
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_paths = []

    for exact_case_name, case_rows in rows_by_exact_case(rows):
        out_dir = case_output_dir(exact_case_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "direct_errors.csv"

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

            for row in case_rows:
                writer.writerow([
                    row["exact_case"],
                    row["anchoring"],
                    row["N"],
                    row["h"],
                    row["L2"],
                    row["H1"],
                    row["vtk_path"] or "",
                ])

        csv_paths.append(csv_path)

    return csv_paths


def write_gradient_comparison(rows):
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_paths = []

    for exact_case_name, case_rows in rows_by_exact_case(rows):
        out_dir = case_output_dir(exact_case_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "gradient_comparison.csv"

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

            for row in case_rows:
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

        csv_paths.append(csv_path)

    return csv_paths


def write_actual_comparison(rows):
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_paths = []

    for row in rows:
        out_dir = case_output_dir(row["exact_case"])
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "actual_comparison.csv"
        energy_history = row["energy_history"]
        update_history = row["update_history"]

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "anchoring",
                "N",
                "h",
                "final_energy",
                "final_update",
                "vtk_path",
            ])
            writer.writerow([
                row["anchoring"],
                row["N"],
                row["h"],
                energy_history[-1] if energy_history else "",
                update_history[-1] if update_history else "",
                row["vtk_path"] or "",
            ])

        csv_paths.append(csv_path)

    return csv_paths


def write_energy_history(exact_case_name, anchoring, energy_history):
    out_dir = case_output_dir(exact_case_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    energy_path = out_dir / f"energy_history_{anchoring}.csv"

    with open(energy_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "energy"])

        for k, energy in enumerate(energy_history):
            writer.writerow([k, energy])

    return energy_path
