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
w_0 = Constant(0)  
w_1 = Constant(10)  
w_2 = Constant(10)  
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

# for the vector (4)
x, y = SpatialCoordinate(mesh)
eps = 1e-3
r2 = x**2 + y**2 + eps
q1_exact = s0 * (x**2 / r2 - 0.5)
q2_exact = s0 * (x * y / r2)
Q_exact = as_tensor([[q1_exact,  q2_exact],
                     [q2_exact, -q1_exact]])





# x = x - 0.5  # for n = (x-0.5, y-0,5)
# y = y - 0.5
# r2 = x**2 + y**2 + 1e-3
# q1_exact = (s0 / 2.0) * (x**2 / r2 - 0.5)
# q2_exact = s0 * x * y / (2*r2)
# Q_exact = as_tensor([[q1_exact, q2_exact],
#                      [q2_exact, -q1_exact]])
# q0 = conditional(ge(x, -1.0), 0.5 * s0, 0.5 * s0)
# q1 = 0.0 * q0
# Q_exact = as_tensor([[q0, q1],
#                      [q1, -q0]])
# n_exact = as_vector([1.0 + 0*x, 0.0 + 0*y]) 
# n_exact = as_vector([cos(pi*x), sin(pi*x)])
# Q_exact = s0 * (outer(n_exact, n_exact) - 0.5 * I)




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

# use USL to solve f and G
f_exact = -l_1 * div(grad(Q_exact)) + bulk_penalty(Q_exact)
G_exact = l_1 * dot(grad(Q_exact), nu) + surface_penalty(Q_exact)
# f_exact = bulk_penalty(Q_exact)
# G_exact = surface_penalty(Q_exact)

# F = LHS - RHS = 0
F_LHS = l_1 * inner(grad(Q), grad(P)) * dx + inner(bulk_penalty(Q), P) * dx + inner(surface_penalty(Q), P) * ds
F_RHS = inner(f_exact, P) * dx + inner(G_exact, P) * ds
F_steady = F_LHS - F_RHS




# direct solver
Q.interpolate(Q_exact)
solve(F_steady == 0, Q, solver_parameters={"snes_linesearch_type": "bt"})
error_L2 = errornorm(Q_exact, Q, norm_type="L2")
error_H1 = errornorm(Q_exact, Q, norm_type="H1")
print(f"Direct Solve MMS L2 Error = {error_L2:.6e}\n")
print(f"Direct Solve MMS H1 Error = {error_H1:.6e}\n")



# gradient descent
dt = Constant(0.0001) 
F_gradient_descent = inner((Q - Q_old) / dt, P) * dx + F_steady
#initial conditoin
# q1_init = 0.05 * (x - 0.5)
# q2_init = 0.05 * (y - 0.5)
# Q.interpolate(as_tensor([
#     [q1_init, q2_init],
#     [q2_init, -q1_init]
# ]))
# Q.interpolate(0.9 * Q_exact)
Q.interpolate(Constant(((0.0, 0.0), (0.0, 0.0))))
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
    (l_1/2.0 * inner(grad(Q), grad(Q)) + bulk_energy(Q) - inner(f_exact, Q)) * dx 
    + (surface_energy(Q) - inner(G_exact, Q)) * ds
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
    current_error = errornorm(Q_exact, Q, norm_type="L2")
    
    print(f"Time t = {t:.2f}, Step Change = {step_diff:.4e}, MMS L2 Error = {current_error:.4e}")
    
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
plt.savefig("energy_descent_(4).png")
plt.close()
print("Saved energy plot to energy_descent.png")

final_error = errornorm(Q_exact, Q, norm_type="L2")
print(f"Final L2 error: {final_error:.6e}")
