from firedrake import *
from firedrake.output import VTKFile
from firedrake.pyplot import tripcolor
import matplotlib.pyplot as plt

mesh_size = 64
# mesh = UnitSquareMesh(mesh_size, mesh_size)
mesh = SquareMesh(mesh_size, mesh_size, L=1.0)
mesh.coordinates.dat.data[:] += 1.0
V = TensorFunctionSpace(mesh, "CG", 2)

Q = Function(V, name="Q")
P = TestFunction(V)

l_1 = Constant(1.0)
l_2 = Constant(0.0)
l_3 = Constant(0.0)
eta = Constant(1.0)
a_0 = Constant(1.0)
a_2 = Constant(7.502104)
a_3 = Constant(60.975813)
a_4 = Constant(66.519069)
s0 = Constant(0.7)

x, y = SpatialCoordinate(mesh)
x_shift = x - 0.5
y_shift = y - 0.5
r2 = x_shift**2 + y_shift**2
q1_exact = (s0 / 2.0) * (x_shift**2 - y_shift**2) / r2
q2_exact = s0 * x_shift * y_shift / r2
Q_exact = as_tensor([[q1_exact, q2_exact],
[q2_exact, -q1_exact]])



bracket_term = (4.0 * l_1 / r2) + (1.0 / eta**2) * (-a_2 + a_4 * (s0**2) / 2.0)
f1 = bracket_term * q1_exact
f2 = bracket_term * q2_exact
f_source = as_tensor([[f1, f2],
                      [f2, -f1]])


F = (
    l_1 * inner(grad(Q), grad(P))
    + l_2 * inner(div(Q), div(P))
    + l_3 * grad(Q)[i, j, k] * grad(P)[i, k, j]
) * dx + (1.0 / eta**2) * (
    - a_2 * inner(Q, P)
    # - a_3 * inner(dot(Q, Q), P)
    + a_4 * inner(Q, Q) * inner(Q, P)
) * dx - inner(f_source, P) * dx

bc = DirichletBC(V, Q_exact, "on_boundary")
Q.interpolate(Q_exact)



# bc_matrix = Constant(((-0.5, 0.6), (0.6, 0.5)))
# bc = DirichletBC(V, bc_matrix, "on_boundary")


solve(F == 0, Q, bcs=bc)
error_L2 = errornorm(Q_exact, Q, norm_type="L2")
print(error_L2)
# print the norm of Q
magnitudevalue = 2*(Q[0,0]**2 + Q[0,1]**2)
V_scalar = FunctionSpace(mesh, "CG", 2)
magnitude_func = Function(V_scalar, name="Magnitude")
magnitude_func.interpolate(magnitudevalue)
print(magnitude_func.dat.data_ro)


# Q_wanted_func = Function(V, name="Q_manufactured").interpolate(Q_exact)
# VTKFile("Q_mms_comparison.pvd").write(Q, Q_wanted_func)

# the magnitude of the Q, which is the constant
q1_num = Q[0, 0]
q2_num = Q[0, 1]
V_scalar = FunctionSpace(mesh, "CG", 2)
S_numerical_expr = sqrt(2.0 * (q1_num**2 + q2_num**2))
S_numerical = Function(V_scalar, name="S_num").project(S_numerical_expr) 
S_manufactured = Function(V_scalar, name="S_exact").interpolate(sqrt(2.0 * (q1_exact**2 + q2_exact**2)))
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig1 = tripcolor(S_numerical, axes=axes[0], cmap="coolwarm", vmin=0.65, vmax=0.75)
axes[0].set_title("Numerical Solution (S)")
axes[0].set_aspect('equal')
fig.colorbar(fig1, ax=axes[0])
fig2 = tripcolor(S_manufactured, axes=axes[1], cmap="coolwarm", vmin=0.65, vmax=0.75)
axes[1].set_title("Manufactured Solution (S)")
axes[1].set_aspect('equal')
fig.colorbar(fig2, ax=axes[1])
plt.suptitle(f"MMS Comparison (L2 Error: {error_L2:.2e})", fontsize=14)
plt.tight_layout()
fig.savefig("2d.png", dpi=300)