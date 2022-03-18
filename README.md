# DWave-Quantum-Annealing
Examples of using DWave quantum annealing system

## How to run this project
First, you should download this repository and then follow the instructions below to install dependencies. Just choose a method you like.

After creating an environment with all dependencies installed, configure your dwave cloud client configuration file according to [Configuring Access to D-Wave Solvers](https://docs.ocean.dwavesys.com/en/stable/overview/sapi.html).

### Use Anaconda
- Install [Anaconda](https://www.anaconda.com/products/individual) and set your preferred shell environment so that you can use the `conda` command.
- Open your preferred shell and change the directory to the repository you downloaded.
- Use `conda env update --prune --file environment.yml` to create a new conda environment and install dependencies. (If you want to create a new dev conda environment, use `conda env update --prune --file environment_dev.yml`)

### Use poetry
- Make sure you have python interpreter in your system.
- Install [poetry](https://python-poetry.org/docs/) and set your preferred shell environment so that you can use the `poetry` command.
- Open your preferred shell and change the directory to the repository you downloaded.
- Use `poetry install --no-dev` to install dependencies. (If you want to install dev dependencies, use `poetry install`)

### Use pip
- Make sure you have python interpreter and pip in your system.
- Open your preferred shell and change the directory to the repository you downloaded.
- Use `pip install -r requirement.txt` to install dependencies. (If you want to install dev dependencies, use `pip install -r requirement_dev.txt`)

## Examples
- Solve Laplace's equation.
  - [Laplace_equation_1D.py](./examples/Laplace_equation_1D.py)
  - [Laplace_equation_1D.ipynb](examples/Laplace_equation_1D.ipynb)
  - [Laplace_equation_2D.py](examples/Laplace_equation_2D.py) 
  - [Laplace_equation_2D.ipynb](examples/Laplace_equation_2D.ipynb)
- Solve Poisson's equation.
  - [Poisson_equation_1D.py](./examples/Poisson_equation_1D.py)
  - [Poisson_equation_1D.ipynb](examples/Poisson_equation_1D.ipynb)
- Solve inverse problem.
  - [inverse_problem_matvec.py](./examples/inverse_problem_matvec.py)
  - [inverse_problem_matvec.ipynb](./examples/inverse_problem_matvec.ipynb)
  - [inverse_problem_poisson1D.py](./examples/inverse_problem_poisson1D.py)
  - [inverse_problem_poisson1D.ipynb](./examples/inverse_problem_poisson1D.ipynb)

**Caution**
`dwaveutils` and `simpeg_ecosys` may frequently change the API in the near future, and the example may crash.

## References
- [ThreeQ.jl](https://github.com/omalled/ThreeQ.jl)
- [QuantumAnnealingInversion.jl](https://github.com/sygreer/QuantumAnnealingInversion.jl)
- [D-Wave System Documentation](https://docs.dwavesys.com/docs/latest/index.html#)
- [D-Wave Ocean Software Documentation](https://docs.ocean.dwavesys.com/en/latest/getting_started.html)
- [Souza, A. M., Cirto, L. J., Martins, E. O., Hollanda, N. L., Roditi, I., Correia, M. D., ... & Oliveira, I. S. (2020). An Application of Quantum Annealing Computing to Seismic Inversion. arXiv preprint arXiv:2005.02846.](https://arxiv.org/abs/2005.02846)
- [Rogers, M. L., & Singleton Jr, R. L. (2019). Floating-point calculations on a quantum annealer: Division and matrix inversion. arXiv preprint arXiv:1901.06526.](https://arxiv.org/abs/1901.06526)
- [Greer, S., & O’Malley, D. (2020). An approach to seismic inversion with quantum annealing. In SEG Technical Program Expanded Abstracts 2020 (pp. 2845-2849). Society of Exploration Geophysicists.](http://www.sygreer.com/research/papers/greer_seisquant_seg_2020.pdf)
