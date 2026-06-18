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

time_step = 1.0e-6
tau = Constant(time_step)

max_iter = 20000
tol = 1.0e-8
save_every = 10


# surface energy parameter
#w0 = Constant(10.0)
#w1 = Constant(0.0)
#w2 = Constant(0.0)

w0 = Constant(0.0)
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

ii,jj,kk = indices(3)
dQdn_exact = as_tensor(Q_exact[ii,jj].dx(kk) *normal[kk],(ii,jj)) #partial v_k Q_exact

surface_linear_exact = ( w0*Q_exact + w1 * (Q_exact-Q_perp(Q_exact)))

surface_nonlinear_exact = surface_w2_term(Q_exact)

boundary_rhs = (
    l1 * dQdn_exact
    + surface_linear_exact
    + surface_nonlinear_exact
)

# GD
Q_new = Q_tensor(q_new)
Q_old = Q_tensor(q)
P = Q_tensor(p)

bulk_new = (
    -a2 * Q_new
    -a3 * dot(Q_new, Q_new)
    + a4 * inner(Q_new, Q_new) * Q_new
)

surface_linear_new = (
    w0 * Q_new
    + w1 * (Q_new - Q_perp(Q_new))
)

surface_nonlinear_new = surface_w2_term(Q_new)

F_newton = (
    inner((Q_new - Q_old) / tau, P) * dx
    + l1 * inner(grad(Q_new), grad(P)) * dx
    + (1.0 / eta**2) * inner(bulk_new, P) * dx
    - inner(f_tensor, P) * dx
    + inner(surface_linear_new + surface_nonlinear_new - boundary_rhs, P) * ds
)

du = TrialFunction(V)
dq = Function(V, name="newton_update")

J_newton = derivative(F_newton, q_new, du)

newton_tol = 1.0e-10
newton_max_it = 25

newton_problem = LinearVariationalProblem(J_newton, -F_newton, dq)
newton_solver = LinearVariationalSolver(
    newton_problem,
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
    q_new.assign(q)

    for m in range(newton_max_it):

        r = assemble(F_newton)

        with r.dat.vec_ro as rv:
            residual_norm = rv.norm()

        if k % 100 == 0:
            print(
                f"timestep {k:5d}, "
                f"Newton iter {m:2d}, "
                f"residual = {residual_norm:.3e}"
            )

        if residual_norm < newton_tol:
            break

        dq.assign(0.0)
        newton_solver.solve()

        q_new.assign(q_new + dq)

    else:
        raise RuntimeError(f"Newton failed at timestep {k}")

    update_norm = math.sqrt(float(assemble(
        inner(Q_tensor(q_new) - Q_tensor(q),
              Q_tensor(q_new) - Q_tensor(q)) * dx
    )))

    q.assign(q_new)

    if k % save_every == 0:
        write_frame(k)

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
