import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config


def plot_direct_errors(rows):
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)

    exact_cases = sorted({row["exact_case"] for row in rows})

    for exact_case_name in exact_cases:
        for anchoring in ["homeotropic", "planar"]:
            data = [
                row
                for row in rows
                if row["exact_case"] == exact_case_name
                and row["anchoring"] == anchoring
            ]

            hs = [row["h"] for row in data]
            L2s = [row["L2"] for row in data]
            H1s = [row["H1"] for row in data]

            plt.figure()
            plt.loglog(hs, L2s, "o-", label="L2 error")
            plt.loglog(hs, H1s, "s-", label="H1 error")
            plt.gca().invert_xaxis()
            plt.xlabel("h")
            plt.ylabel("error")
            plt.title(f"Direct solve MMS errors: {exact_case_name}, {anchoring}")
            plt.legend()
            plt.grid(True, which="both")
            plt.savefig(
                config.OUT_DIR / f"direct_errors_{exact_case_name}_{anchoring}.png",
                dpi=200,
            )
            plt.close()


def plot_energy_history(exact_case_name, anchoring, energy_history):
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(range(len(energy_history)), energy_history)
    plt.xlabel("iteration")
    plt.ylabel("energy")
    plt.title(f"Gradient flow energy: {exact_case_name}, {anchoring}")
    plt.grid(True)
    plt.savefig(
        config.OUT_DIR / f"energy_{exact_case_name}_{anchoring}.png",
        dpi=200,
    )
    plt.close()
