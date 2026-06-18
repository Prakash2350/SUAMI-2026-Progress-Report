from firedrake import *
import matplotlib.pyplot as plt
import numpy as np

# Mesh refinements
Ns = [ 16, 32, 64, 128, 256]

hs = []
errors = []

for N in Ns:
    # 1. Mesh and space
    mesh = UnitSquareMesh(N, N)
    V = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)

    # 2. Trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # 3. Manufactured exact solution
    #u_exact_expr = sin(2*pi*x) * cos(2*pi*y)
    u_exact_expr = sin(x) + x**3

    # Since f = -Delta u_exact
    #f = 8 * pi**2 * sin(2*pi*x) * cos(2*pi*y)
    f = sin(x) - 6*x

    # 4. Variational formulation
    a = dot(grad(u), grad(v)) * dx
    L = f * v * dx

    # 5. Boundary condition g = u_exact on boundary
    bc = DirichletBC(V, u_exact_expr, "on_boundary")

    # 6. Solve
    u_h = Function(V)
    solve(a == L, u_h, bcs=bc)

    # 7. Compute L2 error
    error = errornorm(u_exact_expr, u_h, norm_type="L2")

    # Mesh size h for triangles in UnitSquareMesh(N,N)
    h = np.sqrt(2) / N

    hs.append(h)
    errors.append(error)

# Print table
print(" h              E_h")
print("--------------------------")
for h, e in zip(hs, errors):
    print(f"{h:.6e}   {e:.6e}")

# Compute slope of log-log plot
slope, intercept = np.polyfit(np.log(hs), np.log(errors), 1)
print(f"\nSlope ≈ {slope:.4f}")

# Plot
plt.figure()
plt.loglog(hs, errors, "o-", label=f"Log-Log Error plot; Slope ≈ {slope:.2f}")
plt.xlabel("Mesh size h")
plt.ylabel(r"$E_h = ||u_m - u_h||_{L^2(\Omega)}$")
plt.title("Convergence of FEM for Manufactured Solution")
plt.grid(True, which="both")
plt.legend()
plt.savefig("manufactured_convergence.png", dpi=300)
plt.show()