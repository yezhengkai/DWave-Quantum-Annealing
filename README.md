# DWave-Quantum-Annealing
Examples of using DWave quantum annealing system

## Requirements
To run the examples, you should install the following packages.
- matplotlib
- pandas
- [dwaveutils](https://github.com/yezhengkai/dwaveutils)
- [simpeg-ecosys](https://github.com/yezhengkai/simpeg_ecosys)

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
  - [inverse_problem.py](./examples/inverse_problem.py)
  - [inverse_problem.ipynb](./examples/inverse_problem.ipynb)

**Caution**
`dwaveutils` and `simpeg-ecosys` may frequently change the API in the near future, and the example may crash.

## References
- [ThreeQ.jl](https://github.com/omalled/ThreeQ.jl)
- [QuantumAnnealingInversion.jl](https://github.com/sygreer/QuantumAnnealingInversion.jl)
- [D-Wave System Documentation](https://docs.dwavesys.com/docs/latest/index.html#)
- [D-Wave Ocean Software Documentation](https://docs.ocean.dwavesys.com/en/latest/getting_started.html)
- [Souza, A. M., Cirto, L. J., Martins, E. O., Hollanda, N. L., Roditi, I., Correia, M. D., ... & Oliveira, I. S. (2020). An Application of Quantum Annealing Computing to Seismic Inversion. arXiv preprint arXiv:2005.02846.](https://arxiv.org/abs/2005.02846)
- [Rogers, M. L., & Singleton Jr, R. L. (2019). Floating-point calculations on a quantum annealer: Division and matrix inversion. arXiv preprint arXiv:1901.06526.](https://arxiv.org/abs/1901.06526)
- [Greer, S., & O’Malley, D. (2020). An approach to seismic inversion with quantum annealing. In SEG Technical Program Expanded Abstracts 2020 (pp. 2845-2849). Society of Exploration Geophysicists.](http://www.sygreer.com/research/papers/greer_seisquant_seg_2020.pdf)
