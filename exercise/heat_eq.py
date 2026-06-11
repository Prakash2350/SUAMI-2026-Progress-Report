from firedrake import *
import matplotlib.pyplot as plt
# use the implicit time step via fdm, and fem for the spatial discrete
# f = 0, g = 0, ic is set below


mesh = UnitSquareMesh(40, 40)
V = FunctionSpace(mesh, "CG", 1)

t = 0.0
T_end = 0.1
dt = 0.002 
n_steps = int(T_end / dt)

u = TrialFunction(V)
v = TestFunction(V)

u_old = Function(V)  # u^n
u_next = Function(V) # u^(n+1)
x, y = SpatialCoordinate(mesh)




initial_condition = conditional(
    And(And(x > 0.3, x < 0.7), And(y > 0.3, y < 0.7)), 
    Constant(1.0), 
    Constant(0.0)
)
u_old.interpolate(initial_condition) 
f = Constant(0.0) 
a = u * v * dx + dt * dot(grad(u), grad(v)) * dx
L = u_old * v * dx + dt * f * v * dx
bc = DirichletBC(V, Constant(0.0), "on_boundary")

# only make the graph at time at 0, 12, 25, 49
save_steps = [0, int(n_steps/4), int(n_steps/2), n_steps-1]
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
ax_idx = 0

for step in range(n_steps):
    t += dt
    solve(a == L, u_next, bcs=bc)
    if step in save_steps:
        ax = axes[ax_idx]
        col = tripcolor(u_old, axes=ax, vmin=0, vmax=1.0) 
        ax.set_title(f"Time t = {t:.3f}")
        ax_idx += 1
    u_old.assign(u_next)
plt.tight_layout()
plt.savefig("heat_diffusion.png", dpi=200)