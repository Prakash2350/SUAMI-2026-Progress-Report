import sympy as sp

x, y, s0, l1, eta, a2, a4 = sp.symbols("x y s0 l1 eta a2 a4")

X = x - sp.Rational(1, 2)
Y = y - sp.Rational(1, 2)
r2 = X**2 + Y**2

Q = s0 * (
    sp.Matrix([[X**2, X*Y],
               [X*Y, Y**2]]) / r2
    - sp.eye(2)/2
)

q1 = Q[0, 0]
q2 = Q[0, 1]

normQ2 = sp.simplify(sp.trace(Q * Q))

lapQ = Q.diff(x, 2) + Q.diff(y, 2)

f = sp.simplify(
    -l1 * lapQ
    + (1/eta**2) * (-a2 * Q + a4 * normQ2 * Q)
)

coeff = sp.simplify(
    4*l1/r2 + (1/eta**2) * (-a2 + a4*s0**2/2)
)

print("Q:Q =")
sp.pprint(normQ2)

print("\nCheck ΔQ + (4/r^2)Q =")
sp.pprint(sp.simplify(lapQ + (4/r2)*Q))

print("\nCheck f - coeff Q =")
sp.pprint(sp.simplify(f - coeff*Q))

print("\nf1 =")
sp.pprint(sp.simplify(f[0, 0]))

print("\nf2 =")
sp.pprint(sp.simplify(f[0, 1]))

print("\nCheck f1 = coeff q1:")
print(sp.simplify(f[0, 0] - coeff*q1) == 0)

print("\nCheck f2 = coeff q2:")
print(sp.simplify(f[0, 1] - coeff*q2) == 0)