import matplotlib
from firedrake import *
import matplotlib.pyplot as plt
import numpy as np

N_list = [8, 16, 32, 64, 128]
h_list = []      # for grid size h
error_list = []

for N in N_list:
    mesh = UnitSquareMesh(N, N)
    V = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    h = 1.0 / N 
    
    # u_exact = sin(pi*x) * sin(pi*y)
    # f = 2 * pi**2 * u_exact       
    # g = u_exact 

    u_exact = sin(x) + x**3
    f = sin(x) - 6*x
    g = u_exact

    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u), grad(v)) * dx
    L = f * v * dx
    bc = DirichletBC(V, g, "on_boundary")
    
    u_sol = Function(V)
    solve(a == L, u_sol, bcs=bc)
    
    error = errornorm(u_exact, u_sol, 'L2')
    
    h_list.append(h)
    error_list.append(error)
    print(f"Gird number N={N:3d} | grid size h={h:.4f} | L2 error Eh={error:.6e}")



slope, intercept = np.polyfit(np.log(h_list), np.log(error_list), 1)
plt.figure(figsize=(8, 6))
plt.loglog(h_list, error_list, marker='o', linewidth=2, label=f"Firedrake Error (Slope $\\approx$ {slope:.2f})")
# reference_line = [error_list[0] * (h/h_list[0])**2 for h in h_list]
# plt.loglog(h_list, reference_line, linestyle='--', color='gray', label="Expected $O(h^2)$ Convergence")
plt.xlabel("log mesh size $h$")
plt.ylabel("log $L^2$ Error $E_h$")
plt.title("Log-Log Error Plot")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.savefig("ex3", dpi=200)