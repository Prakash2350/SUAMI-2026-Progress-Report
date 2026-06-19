from firedrake import *
from firedrake.output import VTKFile
import matplotlib.pyplot as plt
import numpy as np

mesh_size = 4
# mesh = SquareMesh(mesh_size, mesh_size, L=1.0)
# V = TensorFunctionSpace(mesh, "CG", 2)

# Q = Function(V, name="Q")
# P = TestFunction(V)
# Q_old = Function(V, name="Q_old")
# x, y = SpatialCoordinate(mesh)

l_1 = Constant(1.0)
eta = Constant(0.1)
a_2 = Constant(7.502104)
a_3 = Constant(60.975813)
a_4 = Constant(66.519069)
s0 = Constant(0.7)
w_0 = Constant(10)  
w_1 = Constant(0)  
w_2 = Constant(0)  
omega = Constant(0.1)

# I = Identity(2)
# nu = FacetNormal(mesh)





# for the circular region
# mesh = SquareMesh(mesh_size, mesh_size, L=2.0)
mesh = UnitDiskMesh(mesh_size)
V = TensorFunctionSpace(mesh, "CG", 2)
Q = Function(V, name="Q")
P = TestFunction(V)
Q_old = Function(V, name="Q_old")
# X = mesh.coordinates
# X.interpolate(as_vector([X[0] - 1.0, X[1] - 1.0]))
# V = TensorFunctionSpace(mesh, "CG", 2)
# Q = Function(V, name="Q")
# P = TestFunction(V)
# Q_old = Function(V, name="Q_old")
# x_old, y_old = X[0], X[1]
# x_new = x_old * sqrt(1.0 - y_old**2 / 2.0)
# y_new = y_old * sqrt(1.0 - x_old**2 / 2.0)
# X.interpolate(as_vector([x_new, y_new]))
I = Identity(2)
nu = FacetNormal(mesh)

# for n0
x, y = SpatialCoordinate(mesh)
n1 = as_vector([x, y])
eps = 1e-3
r2 = x**2 + y**2 + eps
n0 = n1 / sqrt(r2)
Q_init = s0 * (outer(n0, n0) - 0.5 * I)


def bulk_penalty(q):
    return (1.0 / eta**2) * (-a_2 * q - a_3 * dot(q, q) + a_4 * inner(q, q) * q)

def surface_penalty(q):
    q_tilde = q + (s0 / 2.0) * I
    Pi = I - outer(nu, nu)
    q_orth = dot(Pi, dot(q, Pi))
    return (
        w_0 * q 
        + w_1 * (q - q_orth) 
        + (w_2 / omega) * (inner(q_tilde, q_tilde) - s0**2) * q_tilde
    )

F_steady = (
    l_1 * inner(grad(Q), grad(P)) * dx 
    + inner(bulk_penalty(Q), P) * dx 
    + inner(surface_penalty(Q), P) * ds
)



# gradient descent
dt = Constant(0.0001) 
F_gradient_descent = inner((Q - Q_old) / dt, P) * dx + F_steady
Q.interpolate(Q_init)
Q_old.assign(Q) 



def bulk_energy(q):
    return (1.0 / eta**2) * (-a_2/2.0 * inner(q, q) - a_3/3.0 * inner(dot(q, q), q) + a_4/4.0 * inner(q, q)**2)

def surface_energy(q):
    q_tilde = q + (s0 / 2.0) * I
    Pi = I - outer(nu, nu)
    q_orth = dot(Pi, dot(q, Pi))
    return (
        w_0/2.0 * inner(q, q) 
        + w_1/2.0 * inner(q - q_orth, q - q_orth) 
        + (w_2 / (4.0 * omega)) * (inner(q_tilde, q_tilde) - s0**2)**2
    )
Total_Energy_Expr = (
    (l_1 / 2.0 * inner(grad(Q), grad(Q)) + bulk_energy(Q)) * dx 
    + surface_energy(Q) * ds 
)

energy_history = []
time_history = []

t = 0.0
T_max = 2.0 
tol = 1e-5
solver_params = {
    "snes_linesearch_type": "bt",  
    "snes_max_it": 100, 
    "ksp_type": "preonly",  
    "pc_type": "lu" 
}

while t < T_max:
    solve(F_gradient_descent == 0, Q, solver_parameters=solver_params)
    current_energy = assemble(Total_Energy_Expr)
    energy_history.append(current_energy)
    time_history.append(t)
    step_diff = errornorm(Q, Q_old, norm_type="L2")
    
    print(f"Time t = {t:.2f}, Step Change = {step_diff:.4e}")
    
    if step_diff < tol:
        break
        
    Q_old.assign(Q)
    t += float(dt)


plt.figure(figsize=(8, 6))
plt.plot(time_history, energy_history, 'b-', linewidth=2)
plt.xlabel("Time (t)")
plt.ylabel("Total Energy")
plt.title("Energy vs Time during Gradient Descent")
plt.grid(True)
plt.savefig("without_qexact.png")
plt.close()
print("Saved energy plot to energy_descent.png")

