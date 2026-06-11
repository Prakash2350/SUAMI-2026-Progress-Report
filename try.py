from firedrake import *
from firedrake.output import VTKFile
from firedrake.pyplot import tripcolor
import matplotlib.pyplot as plt

mesh_size = 64
mesh = UnitSquareMesh(mesh_size, mesh_size)

V = VectorFunctionSpace(mesh, "CG", 1, dim=2)

q = Function(V, name="q")
p = TestFunction(V)

l_1 = Constant(1.0)
eta = Constant(1.0)
a_2 = Constant(1.0)
a_4 = Constant(1.0)

F = (
    2*l_1*inner(grad(q), grad(p))*dx
    + (2/eta**2)*(-a_2 + 2*a_4*dot(q, q))*dot(q, p)*dx
)

bc = DirichletBC(V, Constant((-0.5, 0.6)), "on_boundary")


solve(F == 0, q, bcs=bc)


VTKFile("q_solution.pvd").write(q)

fig, ax = plt.subplots()
c = tripcolor(q.sub(0), axes=ax)
fig.colorbar(c, ax=ax)
ax.set_title("q1")
fig.savefig("q1.png", dpi=300)
plt.close(fig)

fig, ax = plt.subplots()
c = tripcolor(q.sub(1), axes=ax)
fig.colorbar(c, ax=ax)
ax.set_title("q2")
fig.savefig("q2.png", dpi=300)
plt.close(fig)

