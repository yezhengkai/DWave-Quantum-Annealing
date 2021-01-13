import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
from discretize import TensorMesh
from discretize.utils import mkvc
from dwave.system import DWaveSampler, EmbeddingComposite
from dwaveutils import bl_lstsq
from neal import SimulatedAnnealingSampler
from simpeg_ecosys.mathematical import PoissonCellCentered, VolumeSource

# Define Poisson equation.
# Set the mesh
delta = 1
hx = np.ones(5) * delta
origin = [0]
mesh = TensorMesh([hx], origin=origin)
bc_types = [["dirichlet", "dirichlet"]]
bc_values = [[0, 0]]
print(mesh)

# Set the model parameters (diffusion coefficient)
model_parameters = mkvc(1 * np.ones(mesh.n_cells))  # uniform medium

# Set the source term
source1 = VolumeSource([[mesh.cell_centers_x[(mesh.n_cells//2)]]], values=[1])
source_list = [source1]
print(source_list)

# Construct a PoissonCellCentered instance
poisson = PoissonCellCentered(
    mesh, bc_types, bc_values,
    source_list=source_list,
    model_parameters=model_parameters
)


# Solve poisson's equation by `scipy.sparse.linalg.splu`
# and `scipy.sparse.linagl.spsolve`.
A = poisson.getA().toarray()
b = poisson.getRHS()
print(f"A:\n{A}")
print(f"b:\n{b}")

lu = sp.linalg.splu(A)
x_lu = lu.solve(b)
x = sp.linalg.spsolve(A, b)
if not np.all(np.isclose(x_lu, x)):
    raise ArithmeticError

# Show result
mesh.plot_image(x)
plt.title("Solved by `scipy.sparse.linalg.spsolve`")
print(f"x:\n{x}")


# Solve Poisson's equation using Quantum Annealing (Direct QPU).
# Construct a DirectSolver instance
solver = bl_lstsq.DirectSolver(A, b)

# Set sampler and parameters for DirectSolver.solve method
sampler = EmbeddingComposite(
    DWaveSampler(solver={'qpu': True}, postprocess="sampling")
)  # use postprocess
# sampler = SimulatedAnnealingSampler()
sampling_params = {
    "num_reads": 1000,
    "chain_strength": 100,
    "answer_mode": "histogram"
}
qubo_params = {
    "num_bits": 4,
    "fixed_point": 1,
    "sign": "pn",
    "eq_scaling_val": 1/8,
}

# Solve
result = solver.solve(
    sampler=sampler, sampling_params=sampling_params, qubo_params=qubo_params)
x_directsolver = result["x"]

# Show result
mesh.plot_image(x_directsolver)
plt.title("Solved by `Direct QPU`")
print(f"x:\n{x_directsolver}")


# Solve Poisson's equation using Quantum Annealing (Direct QPU)
# through an iterative procedure.
# Construct a IterativeSolver instance
solver = bl_lstsq.IterativeSolver(A, b)

# Set sampler and parameters for IterativeSolver.solve method
sampler = EmbeddingComposite(
    DWaveSampler(solver={'qpu': True}, postprocess="sampling")
)  # use postprocess
# sampler = SimulatedAnnealingSampler()
sampling_params = {
    "num_reads": 1000,
    "chain_strength": 100,
    "answer_mode": "histogram",
    "postprocess": "sampling"
}
qubo_params = {
    "num_bits": 2,
    "fixed_point": 1,
    "sign": "p",
    "eq_scaling_val": 1/8,
}
iter_params = {
    "scale_factor": 2,
    "num_iter": 20,
    "l2_res_tol": 1e-3,
    "verbose": True
}
rng = np.random.default_rng()
initial_x = (1 - (-1)) * rng.random(b.size) + (-1)

# Solve
result = solver.solve(
    initial_x,
    sampler=sampler,
    sampling_params=sampling_params,
    qubo_params=qubo_params,
    iter_params=iter_params
)
x_iterativesolver = result["x"]

# Show result
mesh.plot_image(x_iterativesolver)
plt.title("Solved by `QA (Direct QPU/iterative)`")
print(f"x:\n{x_iterativesolver}")
plt.show()
