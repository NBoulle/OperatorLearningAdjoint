from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt
from firedrake.preconditioners.patch import bcdofs
from firedrake import *
from slepc4py import SLEPc
from firedrake.petsc import PETSc
import numpy as np
import scipy as sp

# Create mesh
L = 1
W = 0.2
N = 10
mesh = RectangleMesh(int(L / W) * N, N, L, W, quadrilateral=True)

# Define function space
V = VectorFunctionSpace(mesh, "CG", 2)
u = Function(V)
v = TestFunction(V)

# Define Laplacian
def epsilon(u):
    return 0.5 * (grad(u) + grad(u).T)


def residual(u, v):
    a = inner(grad(u), grad(v)) * dx
    return a

# Residual for Laplacian
F = residual(u, v)

# Define boundary conditions
bcs = [DirichletBC(V, Constant((0.0, 0.0)), (1))]

# Solver parameters
params = {
    "ksp_monitor": None,
    "ksp_type": "gmres",
}

# Set up the eigenvalue solver
J = derivative(residual(u, v), u, TrialFunction(V))
A = assemble(J, bcs=bcs)
M = assemble(inner(TestFunction(V), TrialFunction(V)) * dx, bcs=bcs)

# Ensure symmetry of M
for bc in bcs:
    M.M.handle.zeroRowsColumns(bcdofs(bc), diag=0)

# Number of eigenvalues to try
num_eigenvalues = 601

# Compute eigenvalues
# Solver options
opts = PETSc.Options()
opts.setValue("eps_converged_reason", None)
opts.setValue("eps_monitor_conv", None)
opts.setValue("st_type", "sinvert")
opts.setValue("eps_wich", "smallest_magnitude")

# Solve the eigenvalue problem
eps = SLEPc.EPS().create(comm=COMM_WORLD)
eps.setDimensions(num_eigenvalues)
eps.setOperators(A.M.handle, M.M.handle)
eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
eps.setFromOptions()
print("### Solving eigenvalue problem ###")
eps.solve()
eigenfunctions = []
eigenfunction = Function(V, name="Eigenfunction")
eigenvalues = []
for i in range(min(eps.getConverged(), num_eigenvalues)):
    with eigenfunction.dat.vec_wo as x:
        eps.getEigenvector(i, x)
    eigenfunctions.append(eigenfunction.copy(deepcopy=True))
    eigenvalues.append(np.real(eps.getEigenvalue(i)))

# Elasticity equations
u = Function(V)
u.rename("Solution")
v = TestFunction(V)
rhs = Function(V)
Id = Identity(mesh.geometric_dimension())  # 2x2 Identity tensor
mu = 1
lambda_ = Constant(1.25)  # 1.25


def sigma(u):
    return lambda_ * div(u) * Id + 2 * mu * epsilon(u)


uh = TrialFunction(V)
F = inner(sigma(uh), epsilon(v)) * dx
LHS = dot(rhs, v) * dx
delta = W / L
gamma = 0.4 * delta**2
g = gamma
f = Constant((0.0, - g))
rhs.assign(f)
solve(F == LHS, u, bcs=bcs, solver_parameters=params)

# Get the mass matrix
M = assemble(inner(TestFunction(V), TrialFunction(V)) * dx, bcs=bcs)
A = sp.sparse.csr_matrix(M.M.handle.getValuesCSR()[::-1])

# Solve for each eigenfunction
Mn = np.empty((2 * u.dat.data.shape[0], 0))
Rn = np.empty((2 * u.dat.data.shape[0], 0))
for k in range(num_eigenvalues):
    rhs.assign(eigenfunctions[k])
    solve(F == LHS, u, bcs=bcs, solver_parameters=params)
    Mn = np.hstack((Mn, np.reshape(u.dat.data, (-1, 1), order='C')))
    Rn = np.hstack(
        (Rn, eigenvalues[k] * np.reshape(u.dat.data, (-1, 1), order='C')))

# Compute ||A||
U, S, V = randomized_svd(Mn, n_components=1)
# Normalize singular vector wrt mass matrix
norm_A = S[0] * np.sqrt(np.inner(U[:, 0].flatten(), A.dot(U[:, 0]).flatten()))
print("|A| = %.2e" % (norm_A))

# Select column in Mn
norm_E = []
for s in range(1, num_eigenvalues):
    Mn_s = Mn[:, s:]
    U, S, V = randomized_svd(Mn_s, n_components=1)
    # Normalize singular vector wrt mass matrix
    s1 = S[0] * np.sqrt(np.inner(U[:, 0].flatten(), A.dot(U[:, 0]).flatten()))
    norm_E += [s1]
    print("s = %d: |A-APs| = %.2e" % (s, norm_E[-1]))

# Compute RHS
norm_ALP = []
for n in range(num_eigenvalues):
    U, S, V = randomized_svd(Rn[:, :n + 1], n_components=1)
    c = S[0] * np.sqrt(np.inner(U[:, 0].flatten(), A.dot(U[:, 0]).flatten()))
    norm_ALP += [c]

c = norm_ALP[-1]
print("|ALPn| = %.2e" % c)

RHS = [c / (norm_A * eigenvalues[k + 1]) for k in range(len(norm_E))]
norm_E = [s / norm_A for s in norm_E]

# Save data
np.savetxt("elasticity_2d_result.csv", np.vstack(
    (range(1, num_eigenvalues), np.array(norm_E), np.array(RHS))).T, delimiter=",")
np.savetxt("elasticity_2d_Mn.csv", np.vstack(
    (range(1, num_eigenvalues + 1), np.array(norm_ALP))).T, delimiter=",")

plt.subplot(1, 2, 1)
plt.loglog(range(1, num_eigenvalues), norm_E)
plt.loglog(range(1, num_eigenvalues), RHS)
plt.xlim([1, 400])

plt.subplot(1, 2, 2)
plt.plot(range(1, num_eigenvalues + 1), norm_ALP)
plt.savefig("fig_eig_%d_lmbda_%.2e_N_%d.png" %
            (num_eigenvalues, float(lambda_), N))
