from firedrake import *
import matplotlib.pyplot as plt
from firedrake.pyplot import tripcolor

mesh = UnitSquareMesh(32,32)
V = FunctionSpace(mesh, "CG", 1)
x, y = SpatialCoordinate(mesh)

u = TrialFunction(V)
v = TestFunction(V)

f = Constant(0.0)
# g = Constant(0.0) 
g = x**2 + y**2

a = dot(grad(u), grad(v)) * dx



# cases = [
#     (Constant(0.0), Constant(0.0), "aa"),
#     (Constant(0.0), x**2 + y**2, "ab"),
#     (Constant(0.0), sin(2*pi*x)*sin(2*pi*y), "ac"),
#     (x**2 + y**2, Constant(0.0), "ba"),
#     (x**2 + y**2, x**2 + y**2, "bb"),
#     (x**2 + y**2, sin(2*pi*x)*sin(2*pi*y), "bc"),
#     (sin(2*pi*x)*sin(2*pi*y), Constant(0.0), "ca"),
#     (sin(2*pi*x)*sin(2*pi*y), x**2 + y**2, "cb"),
#     (sin(2*pi*x)*sin(2*pi*y), sin(2*pi*x)*sin(2*pi*y), "cc"),
# ]

cases = [
    (Constant(1.0), Constant(0.0), "aa"),
    (Constant(1.0), x**2 + y**2, "ab"),
    (Constant(1.0), sin(2*pi*x)*cos(2*pi*y), "ac"),
    (x**2 + y**2, Constant(0.0), "ba"),
    (x**2 + y**2, x**2 + y**2, "bb"),
    (x**2 + y**2, sin(2*pi*x)*cos(2*pi*y), "bc"),
    (sin(2*pi*x)*cos(2*pi*y), Constant(0.0), "ca"),
    (sin(2*pi*x)*cos(2*pi*y), x**2 + y**2, "cb"),
    (sin(2*pi*x)*cos(2*pi*y), sin(2*pi*x)*cos(2*pi*y), "cc"),
]

fig, axes = plt.subplots(3, 3, figsize=(15, 14))
axes_flat = axes.flatten()
u_sol = Function(V)

for i, (f_expr, g_expr, title) in enumerate(cases):
    L = f_expr * v * dx
    bc = DirichletBC(V, g_expr, "on_boundary")

    solve(a == L, u_sol, bcs=bc)
    current_ax = axes_flat[i]
    
    collection = tripcolor(u_sol, axes=current_ax)
    fig.colorbar(collection, ax=current_ax)
    current_ax.set_title(title)

plt.tight_layout()
plt.savefig("ex2", dpi=200)