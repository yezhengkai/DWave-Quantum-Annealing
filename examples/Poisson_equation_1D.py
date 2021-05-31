import warnings
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

# Suppress all warnings
warnings.filterwarnings('ignore')

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

# Create an instance of `PoissonCellCentered`
poisson = PoissonCellCentered(
    mesh, bc_types, bc_values,
    source_list=source_list,
    model_parameters=model_parameters
)


# Solve Poisson equation by `scipy.sparse.linalg.splu` and `scipy.sparse.linagl.spsolve`.
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
print(f"x_spsolve:\n{x}")


# Solve Poisson equation using Simulated Annealing or Quantum Annealing (Direct QPU)
# Create an instance of `BlLstsqProblem`
problem_params = {"A": A, "b": b}
qubo_params = {
    "num_bits": 4,
    "fixed_point": 1,
    "sign": "pn",
    "eq_scaling_val": 1/8,
}
problem = bl_lstsq.problem.BlLstsqProblem(problem_params, qubo_params)

# Create an instance of `BlLstsqDirectSolver`
solver = bl_lstsq.solver.BlLstsqDirectSolver(problem)

# Set sampler and parameters for `BlLstsqDirectSolver.solve` method
sampler = SimulatedAnnealingSampler()
# sampler = EmbeddingComposite(
#     DWaveSampler(solver={'qpu': True}, postprocess="sampling")
# )  # use postprocess
sampling_params = {
    "num_reads": 1000,
    "chain_strength": 100,  # no effect on SA
    "answer_mode": "histogram"
}

# Solve
result = solver.solve(sampler=sampler, sampling_params=sampling_params)
x_directsolver = result["x"]

# Show result
mesh.plot_image(x_directsolver)
plt.title("Solved by `SA`")
# plt.title("Solved by `QA (Direct QPU)`")
print(f"x_directsolver:\n{x_directsolver}")


# Solve Poisson equation using Simulated Annealing or Quantum Annealing (Direct QPU) through an iterative procedure
# Create an instance of `BlLstsqProblem`
problem_params = {"A": A, "b": b}
qubo_params = {
    "num_bits": 2,
    "fixed_point": 1,
    "sign": "p",
    "eq_scaling_val": 1/8,
}
problem = bl_lstsq.problem.BlLstsqProblem(problem_params, qubo_params)

# Create an instance of `BlLstsqIterativeSolver`
solver = bl_lstsq.BlLstsqIterativeSolver(problem)

# Set initial x, sampler and parameters for `BlLstsqIterativeSolver.solve` method
sampler = SimulatedAnnealingSampler()
# sampler = EmbeddingComposite(
#     DWaveSampler(solver={'qpu': True}, postprocess="sampling")
# )  # use postprocess
sampling_params = {
    "num_reads": 1000,
    "chain_strength": 100,  # no effect on SA
    "answer_mode": "histogram"
}
iter_params = {
    "scale_factor": 2,
    "num_iter": 20,
    "obj_tol": 1e-3,
    "verbose": True
}
rng = np.random.default_rng(1234)
initial_x = (1 - (-1)) * rng.random(b.size) + (-1)

# Solve
result = solver.solve(
    initial_x,
    sampler=sampler,
    sampling_params=sampling_params,
    iter_params=iter_params
)
x_iterativesolver = result["x"]

# Show result
mesh.plot_image(x_iterativesolver)
plt.title("Solved by `SA` (iterative)")
# plt.title("Solved by `QA (Direct QPU/iterative)`")
print(f"x_iterativesolver:\n{x_iterativesolver}")
plt.show()
