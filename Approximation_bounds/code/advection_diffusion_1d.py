from firedrake.preconditioners.patch import bcdofs
from firedrake import *
from slepc4py import SLEPc
from firedrake.petsc import PETSc
import numpy as np
import scipy as sp
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt

# Create mesh
N = 1000
mesh = UnitIntervalMesh(N)

# Define function space
V = FunctionSpace(mesh, "CG", 3)
u = Function(V)
v = TestFunction(V)


def residual(u, v):
    a = inner(grad(u), grad(v)) * dx
    return a


F = residual(u, v)

# Define boundary conditions
bcs = [DirichletBC(V, Constant(0.0), (1, 2))]

# Solver parameters
params = {
    "snes_max_it": 100,
    "snes_atol": 1.0e-9,
    "snes_rtol": 0.0,
    "snes_linesearch_type": "basic",
    "ksp_type": "preonly",
    "pc_type": "lu"
}

# Solve problem
solve(F == 0, u, bcs=bcs, solver_parameters=params)

# Set up the eigenvalue solver
J = derivative(residual(u, v), u, TrialFunction(V))
A = assemble(J, bcs=bcs)
M = assemble(inner(TestFunction(V), TrialFunction(V)) * dx, bcs=bcs)

# There must be a better way of doing this
for bc in bcs:
    # Ensure symmetry of M
    M.M.handle.zeroRowsColumns(bcdofs(bc), diag=0)

# Number of eigenvalues to try
num_eigenvalues = 601

# Compute eigenvalues
# Solver options
opts = PETSc.Options()
opts.setValue("eps_converged_reason", None)
# opts.setValue("eps_monitor_conv", None)
opts.setValue("st_type", "sinvert")
opts.setValue("eps_wich", "smallest_magnitude")
opts.setValue("eps_conv_norm", None)

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

u = Function(V)
v = TestFunction(V)
rhs = Function(V)
F = inner(-0.25 * grad(u), grad(v)) * dx + \
    inner(5*Dx(u, 0) + u, v) * dx - inner(rhs, v) * dx

# Solve for each eigenfunction
Mn = np.empty((u.dat.data.shape[0], 0))
Rn = np.empty((u.dat.data.shape[0], 0))
for k in range(num_eigenvalues):
    rhs.assign(eigenfunctions[k])
    solve(F == 0, u, bcs=bcs, solver_parameters=params)
    Mn = np.hstack((Mn, np.reshape(u.dat.data, (-1, 1))))
    Rn = np.hstack((Rn, eigenvalues[k] * np.reshape(u.dat.data, (-1, 1))))

# Compute mass matrix
M = assemble(inner(TestFunction(V), TrialFunction(V)) * dx, bcs=bcs)
A = sp.sparse.csr_matrix(M.M.handle.getValuesCSR()[::-1])


# Compute norm of A
U, S, V = randomized_svd(Mn, n_components=1)
# Normalize singular vector wrt mass matrix
norm_A = S[0] * np.sqrt(np.inner(U[:, 0], A.dot(U[:, 0])))
print("|A| = %.2e" % norm_A)

# Select column in Mn
norm_E = []
for s in range(1, num_eigenvalues):
    Mn_s = Mn[:, s:]
    U, S, V = randomized_svd(Mn_s, n_components=1)
    # Normalize singular vector wrt mass matrix
    s1 = S[0] * np.sqrt(np.inner(U[:, 0], A.dot(U[:, 0])))
    norm_E += [s1]
    print("s = %d: |A-APs| = %.2e" % (s, norm_E[-1]))

# Compute RHS
norm_ALP = []
for n in range(num_eigenvalues):
    U, S, V = randomized_svd(Rn[:, :n + 1], n_components=1)
    c = S[0] * np.sqrt(np.inner(U[:, 0], A.dot(U[:, 0])))
    norm_ALP += [c]

c = norm_ALP[-1]
print("|ALPn| = %.2e" % c)

RHS = [c / (norm_A * eigenvalues[k + 1]) for k in range(len(norm_E))]
norm_E = [norm_E[k] / norm_A for k in range(len(norm_E))]

# Save data
np.savetxt("advection_diffusion_1d_result.csv", np.vstack(
    (range(1, num_eigenvalues), np.array(norm_E), np.array(RHS))).T, delimiter=",")
np.savetxt("advection_diffusion_1d_Mn.csv", np.vstack(
    (range(1, num_eigenvalues + 1), np.array(norm_ALP))).T, delimiter=",")

plt.subplot(1, 2, 1)
plt.loglog(range(1, num_eigenvalues), norm_E)
plt.loglog(range(1, num_eigenvalues), RHS)

plt.subplot(1, 2, 2)
plt.semilogx(range(1, num_eigenvalues + 1), norm_ALP)
plt.show()
