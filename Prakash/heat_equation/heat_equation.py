from firedrake import *
import matplotlib.pyplot as plt

# 1. Mesh and space
mesh = UnitSquareMesh(40, 40)
V = FunctionSpace(mesh, "CG", 1)
x, y = SpatialCoordinate(mesh)

# 2. Time variables
t = 0.0
T_end = 0.1
dt = 0.002
n_steps = int(T_end / dt)

# 3. Trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# 4. Initial condition: Gaussian heat source
u_old = Function(V)
u_old.interpolate(
    exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.02)
)

# 5. Source term
f = Constant(0.0)

# 6. Variational formulation
a = u * v * dx + dt * dot(grad(u), grad(v)) * dx
L = u_old * v * dx + dt * f * v * dx

# 7. Zero Dirichlet boundary condition
bc = DirichletBC(V, Constant(0.0), "on_boundary")

# 8. Create 3x3 figure
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.flatten()

# Save 9 equally spaced snapshots
snapshot_steps = [
    int(k * (n_steps - 1) / 8)
    for k in range(9)
]

# 9. Time stepping
u_next = Function(V)
plot_number = 0

for step in range(n_steps):

    solve(a == L, u_next, bcs=bc)

    u_old.assign(u_next)
    t += dt

    if step in snapshot_steps:

        ax = axes[plot_number]

        colors = tripcolor(
            u_next,
            axes=ax
        )

        fig.colorbar(
            colors,
            ax=ax
        )

        ax.set_title(f"t = {t:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        plot_number += 1

# 10. Final formatting
plt.suptitle(
    "Heat Equation with Gaussian Initial Condition\n"
    "Zero Dirichlet Boundary Condition",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    "heat_evolution_3x3.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()