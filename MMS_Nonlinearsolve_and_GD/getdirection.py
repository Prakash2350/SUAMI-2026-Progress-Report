from firedrake import *
from firedrake.output import VTKFile
from firedrake.pyplot import tripcolor
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri

mesh_size = 64
# mesh = UnitSquareMesh(mesh_size, mesh_size)
mesh = SquareMesh(mesh_size, mesh_size, L=1.0)
# mesh.coordinates.dat.data[:] += 1.0
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
r2 = x_shift**2 + y_shift**2 + 1e-3
q1_exact = (s0 / 2.0) * (x_shift**2 - y_shift**2) / r2
q2_exact = s0 * x_shift * y_shift / r2 
Q_exact = as_tensor([[q1_exact, q2_exact],
[q2_exact, -q1_exact]])


# Q = Function(V, name="Q")
# Q.interpolate(Q_exact)



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

solve(F == 0, Q, bcs=bc)
#finish for calculating the Q




# figure out the direction n
# V_vec = VectorFunctionSpace(mesh, "CG", 2)
# n_func = Function(V_vec, name="Director")
# Q_data = Q.dat.data_ro  # has #nodes elements, with each elements is a 2×2 Q-tensor 
# n_data = n_func.dat.data 

# for i in range(len(Q_data)):
#     Q_mat = Q_data[i]
#     eigvals, eigvecs = np.linalg.eigh(Q_mat)
#     max_idx = np.argmax(eigvals)
#     n_vec = eigvecs[:, max_idx]
#     n_data[i, 0] = n_vec[0] 
#     n_data[i, 1] = n_vec[1]


V_scalar = FunctionSpace(mesh, "CG", 2)
q00_func = project(Q[0, 0], V_scalar)
q01_func = project(Q[0, 1], V_scalar)
q10_func = project(Q[1, 0], V_scalar)
q11_func = project(Q[1, 1], V_scalar)
q00_data = q00_func.dat.data_ro
q01_data = q01_func.dat.data_ro
q10_data = q10_func.dat.data_ro
q11_data = q11_func.dat.data_ro
V_vec = VectorFunctionSpace(mesh, "CG", 2)
n_func = Function(V_vec, name="Director")
n_data = n_func.dat.data 
# for i in range(len(q00_data)):
#     Q_mat = np.array([[q00_data[i], q01_data[i]],
#                       [q10_data[i], q11_data[i]]])
#     eigvals, eigvecs = np.linalg.eigh(Q_mat)
#     max_idx = np.argmax(eigvals)
#     n_vec = eigvecs[:, max_idx]
#     n_data[i, 0] = n_vec[0] 
#     n_data[i, 1] = n_vec[1]


V_coord = VectorFunctionSpace(mesh, "CG", 2) 
coords_func = Function(V_coord).interpolate(SpatialCoordinate(mesh))
mesh_coords = coords_func.dat.data_ro
for i in range(len(q00_data)):
    Q_mat = np.array([[q00_data[i], q01_data[i]],
                      [q10_data[i], q11_data[i]]])
    eigvals, eigvecs = np.linalg.eigh(Q_mat)
    max_idx = np.argmax(eigvals)
    n_vec = eigvecs[:, max_idx]
    x_coord = mesh_coords[i, 0] 
    y_coord = mesh_coords[i, 1]
    r_vec = np.array([x_coord - 0.5, y_coord - 0.5])
    # if the angle between the eigenvector and radial direction, then rotate it
    if np.dot(n_vec, r_vec) < 0:
        n_vec = -n_vec
    n_data[i, 0] = n_vec[0] 
    n_data[i, 1] = n_vec[1]







# for the graph
V_coord = VectorFunctionSpace(mesh, "CG", 2)
coord_func = Function(V_coord).interpolate(SpatialCoordinate(mesh))
coords = coord_func.dat.data_ro 
x_coords = coords[:, 0]
y_coords = coords[:, 1]
n_data = n_func.dat.data_ro
nx = n_data[:, 0]
ny = n_data[:, 1]
skip = 10
X = x_coords[::skip]
Y = y_coords[::skip]
U = nx[::skip]
V = ny[::skip]
plt.figure(figsize=(6, 6))
plt.quiver(X, Y, U, V, 
           color='blue', 
           scale=45, 
           pivot='middle', 
           headwidth=3, 
           headlength=4)

plt.gca().set_aspect("equal")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.title("Director Vector Graph (n)")
plt.tight_layout()
output_file = "director_vector_graph.png"
plt.savefig(output_file, dpi=300)
plt.close()