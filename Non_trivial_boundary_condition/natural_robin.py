import math 
import numpy as np
from pathlib import Path

from firedrake import *
from firedrake.output import VTKFile

CODE_DIR = Path(__file__).resolve().parent

#   Mesh
mesh = UnitSquareMesh(64,64)

V = VectorFunctionSpace(mesh,"CG",1,dim = 2)
S = FunctionSpace(mesh,"CG",1)

q = Function(V,name = "q")
q_new = Function(V,name = "q_new")

u = TrialFunction(V)
p = TestFunction(V)

x,y = SpatialCoordinate(mesh)


#   Parameters
s0 = Constant(0.7)

l1 = Constant(1.0)
eta = Constant(1)

a2 = Constant(7.5)
a3 = Constant(61.0)
a4 = Constant(66.52)

eps = Constant(1.0e-4)

time_step = 1.0e-4
tau = Constant(time_step)

max_iter = 20000
tol = 1.0e-10
save_every = 10


# surface energy parameter
#w0 = Constant(10.0)
#w1 = Constant(0.0)
#w2 = Constant(0.0)

w0 = Constant(0,0)
w1 = w2 = Constant(10.0)

omega = Constant(0.1)

use_mms_robin = True

normal = FacetNormal(mesh) # mu 
I2 = Identity(2)

#tracelss Q-tensor
def Q_tensor(v):
    return as_tensor([[v[0],  v[1]],[v[1], -v[0]],])

#boundary tensor operations
def Q_perp(Q):
    Pi = I2 - outer(normal,normal)
    return dot(Pi,dot(Q,Pi)) 

def Q_tilde(Q):
    return Q + (s0/2) * I2

def surface_w2_term(Q):
    Qt = Q_tilde(Q)
    return (w2 / omega) * (inner(Qt, Qt) - s0**2) * Qt

Q_gamma = s0 * (outer(normal,normal) - 0.5*I2)


# Manufactured exact solution
X = x - 0.5
Y = y - 0.5
r2 = X**2 + Y**2 + eps

q_exact = as_vector([
    (s0 / 2.0) * (X**2 - Y**2) / r2,
    s0 * X * Y / r2,
])

Q_exact = Q_tensor(q_exact)

f_tensor = (
    -l1 * div(grad(Q_exact))
    + (1.0 / eta**2) * (
        -a2 * Q_exact
        -a3 * dot(Q_exact, Q_exact)
        + a4 * inner(Q_exact, Q_exact) * Q_exact
    )
)

boundary_rhs = (
    w0 * Q_gamma
    - (w1 * s0 / 2.0) * outer(normal, normal)
)

# GD
q.interpolate(as_vector([0.0,0.0,]))

Q_trial = Q_tensor(u)
Q_old = Q_tensor(q)
P = Q_tensor(p)


bulk_old = (
    -a2 * Q_old
    -a3 * dot(Q_old, Q_old)
    + a4 * inner(Q_old, Q_old) * Q_old
)

surface_linear_trial = (
    w0 * Q_trial
    + w1 * (Q_trial - Q_perp(Q_trial))
)

surface_nonlinear_old = surface_w2_term(Q_old)

A = (
    inner(Q_trial / tau, P) * dx
    + l1 * inner(grad(Q_trial), grad(P)) * dx
    + inner(surface_linear_trial, P) * ds
)

L = (
    inner(Q_old / tau, P) * dx
    - (1.0 / eta**2) * inner(bulk_old, P) * dx
    + inner(f_tensor, P) * dx
    + inner(boundary_rhs - surface_nonlinear_old, P) * ds
)


problem = LinearVariationalProblem(A,L,q_new)
solver = LinearVariationalSolver(
    problem,
    solver_parameters={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
)


# director
W = VectorFunctionSpace(mesh, "CG", 1, dim=3)

director = Function(W, name="director")

director_file = VTKFile(str(CODE_DIR / "director_robin_time.pvd"))


def update_director():
    q_vals = q.dat.data_ro
    d_vals = director.dat.data

    for node in range(len(q_vals)):
        q1 = q_vals[node, 0]
        q2 = q_vals[node, 1]

        Qmat = np.array([
            [q1,  q2],
            [q2, -q1],
        ])

        eigvals, eigvecs = np.linalg.eigh(Qmat)

        # director = eigenvector of largest eigenvalue
        n = eigvecs[:, np.argmax(eigvals)]

        d_vals[node, 0] = n[0]
        d_vals[node, 1] = n[1]
        d_vals[node, 2] = 0.0

def write_frame(k):
    t = k * time_step
    update_director()
    director_file.write(director, time=t)


write_frame(0)
for k in range(1, max_iter + 1):

    solver.solve()

    update_norm = math.sqrt(float(assemble(
        inner(Q_tensor(q_new) - Q_tensor(q),
              Q_tensor(q_new) - Q_tensor(q)) * dx
    )))

    q.assign(q_new)

    if k % save_every == 0:
        write_frame(k)

    if k % 100 == 0:
        error_L2 = math.sqrt(float(assemble(
            inner(Q_tensor(q) - Q_exact, Q_tensor(q) - Q_exact) * dx
        )))

        print(
            f"iter = {k:5d}, "
            f"time = {k * time_step:.6e}, "
            f"update = {update_norm:.3e}, "
            f"L2 error vs Q_exact = {error_L2:.3e}"
        )

    if update_norm < tol:
        print(f"Converged at iter = {k}, update = {update_norm:.3e}")
        write_frame(k)
        break


print(
    "Saved in:\n"
    f"  {CODE_DIR}\n\n"
    "File:\n"
    "  director_robin_time.pvd"
)
