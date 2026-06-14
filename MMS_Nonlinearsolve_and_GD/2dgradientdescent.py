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


Q_old = Function(V, name="Q_old")
dt = Constant(0.01) 
F_gradient_descent = inner((Q - Q_old) / dt, P) * dx + F
Q.interpolate(Constant(((0.0, 0.0), (0.0, 0.0))))
Q_old.assign(Q) 


t = 0.0
T_max = 2.0 
tol = 1e-5 

while t < T_max:
    solve(F_gradient_descent == 0, Q, bcs=bc)
    diff = errornorm(Q, Q_old, norm_type="L2")
    print(f"Time t = {t:.2f}, Step Change = {diff:.4e}")
    
    if diff < tol:
        break
    Q_old.assign(Q)
    t += float(dt)

error_L2 = errornorm(Q_exact, Q, norm_type="L2")
print(error_L2)


# print Q[0,0] and Q[0,1]
V_scalar = FunctionSpace(mesh, "CG", 2)
q00_func = Function(V_scalar, name="q00")
q00_func.interpolate(Q[0, 0])
q01_func = Function(V_scalar, name="q01")
q01_func.interpolate(Q[0, 1])
print(q00_func.dat.data_ro)
print(q01_func.dat.data_ro)
print("end")