"""binary linear least square."""

import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict

import numpy as np
from scipy.sparse import diags, dok_matrix
from tqdm import tqdm


def discretize_matrix(matrix, bit_value):
    return np.kron(matrix, bit_value)


def get_bit_value(num_bits, fixed_point=0, sign='pn'):
    """The value of each bit in two's-complement binary fixed-point numbers."""
    # 'pn': positive and negative value
    # 'p': only positive value
    # 'n': only negative value
    accepted_sign = ['pn', 'p', 'n']
    if sign not in accepted_sign:
        warnings.warn('Use default `sign` setting.')
        sign = 'pn'

    if sign == 'pn':
        return np.array([-2 ** fixed_point if i == 0
                         else 2. ** (fixed_point - i)
                         for i in range(0, num_bits)])
    elif sign == 'p':
        return np.array([2. ** (fixed_point - i)
                         for i in range(1, num_bits + 1)])
    else:
        return np.array([-2. ** (fixed_point - i)
                         for i in range(1, num_bits + 1)])


def bruteforce(A_discrete, b, bit_value):
    """Solve A_discrete*q=b where q is a binary vector by brute force."""

    # number of predictor
    num_predictor_discrete = A_discrete.shape[1]
    # total number of solutions
    num_solution = 2 ** num_predictor_discrete
    # initialize minimum 2-norm
    min_norm = np.inf

    # loop all solutions
    with tqdm(range(num_solution), desc='brute force') as pbar:
        for i in pbar:
            # assign solution
            # https://stackoverflow.com/questions/13557937/how-to-convert-decimal-to-binary-list-in-python/13558001
            # https://stackoverflow.com/questions/13522773/convert-an-integer-to-binary-without-using-the-built-in-bin-function
            q = np.array([int(bit)
                          for bit in format(i, f'0{num_predictor_discrete}b')])
            # calculate 2-norm
            new_norm = np.linalg.norm(A_discrete @ q - b, 2)
            # update best solution
            if new_norm < min_norm:
                min_norm = new_norm
                best_q = np.copy(q)

    if 'best_q' in locals():
        best_x = q2x(best_q, bit_value)
        return best_q, best_x, min_norm
    else:
        raise NameError("name 'best_q' is not defined")


def q2x(q, bit_value):
    """Convert vector of bit to vector of real value."""

    if not isinstance(q, np.ndarray):
        raise TypeError(
            f"q must be the instance of np.ndarray, not {type(q)}"
        )
    if not isinstance(bit_value, np.ndarray):
        raise TypeError(
            f"bit_value must be the instance of np.ndarray, not {type(bit_value)}"
        )

    q, bit_value = q.flatten(), bit_value.flatten()
    num_q_entry = len(q)
    num_bits = len(bit_value)
    num_x_entry = num_q_entry // num_bits

    if num_x_entry * num_bits != num_q_entry:
        raise ValueError('The length of q or bit_value is incorrect.')

    x = np.array(
        [
            bit_value @ q[i*num_bits:(i+1)*num_bits]
            for i in range(num_x_entry)
        ]
    )
    return x


def get_qubo(A_discrete, b, eq_scaling_val=1 / 8):
    """
    Get coefficients of a quadratic unconstrained binary optimization (QUBO)
    problem defined by the dictionary.
    """

    # define weights
    # https://stackoverflow.com/questions/37524151/convert-a-deafultdict-to-numpy-matrix-or-a-csv-of-2d-matrix
    # https://scipy-lectures.org/advanced/scipy_sparse/dok_matrix.html
    qubo_a = (np.diag(A_discrete.T @ A_discrete)
              - 2 * A_discrete.T @ b.flatten())
    qubo_b = dok_matrix(np.tril(2 * A_discrete.T @ A_discrete, k=-1))

    # define objective
    Q = defaultdict(
        int,
        (eq_scaling_val * (diags(qubo_a, format='dok') + qubo_b)).items()
    )

    # force the diagonal entries to have values
    for i in range(qubo_a.size):
        Q[(i, i)] += 0.0

    return Q


class BaseSolver(metaclass=ABCMeta):

    required_qubo_params = [
        "num_bits",
        "fixed_point",
        "sign"
    ]

    def __init__(self, A, b,
                 sampler=None,
                 sampling_params=None,
                 qubo_params=None) -> None:

        if sampler is not None:
            self._check_sampler(sampler)
        if qubo_params is not None:
            self._check_qubo_params(qubo_params)

        super().__init__()
        self._A = A
        self._b = b
        self.sampler = sampler
        self.sampling_params = sampling_params
        self.qubo_params = qubo_params

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, val):
        if val.shape[0] != self._b.size:
            raise ValueError(
                "The number of rows of matrix `A` must be the same as"
                " the length of the vector `b`."
            )
        self._A = val

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, val):
        if val.size != self._A.shape[0]:
            raise ValueError(
                "The length of the vector `b` must be the same as"
                " the number of rows of matrix `A`."
            )
        self._b = val

    def _check_sampler(self, sampler):
        if not hasattr(sampler, "sample_qubo"):
            raise AttributeError(
                f"Sampler must have the `sampler_qubo` method.")

    def _check_qubo_params(self, qubo_params):
        if not isinstance(qubo_params, dict):
            raise TypeError(
                f"qubo_params must be a dict, not {type(qubo_params)}"
            )
        for param in self.required_qubo_params:
            if param not in qubo_params:
                raise KeyError(f"{param} must in `qubo_params`")

    def _assign(self, params, attr, default_value=None, check_func=None):
        if params is None \
                and getattr(self, attr, None) is None:
            if default_value is not None:
                setattr(self, attr, default_value)
                warnings.warn(f"Use default `{attr}`: {default_value}")
            else:
                raise ValueError(f"Please enter a `{attr}`.")
        elif params is None \
                and getattr(self, attr, None) is not None:
            pass
        else:
            if check_func is not None and callable(check_func):
                check_func(params)
            setattr(self, attr, params)

    @abstractmethod
    def solve(self):
        pass


class DirectSolver(BaseSolver):

    def __init__(self, A, b,
                 sampler=None,
                 sampling_params=None,
                 qubo_params=None) -> None:

        super().__init__(A, b, sampler=sampler,
                         sampling_params=sampling_params,
                         qubo_params=qubo_params)

    def solve(self,
              sampler=None,
              sampling_params=None,
              qubo_params=None):

        self._assign(sampler, "sampler")
        self._assign(sampling_params, "sampling_params")
        self._assign(
            qubo_params, "qubo_params",
            default_value=self.default_qubo_params(),
            check_func=self._check_qubo_params
        )

        # get qubo
        bit_value = get_bit_value(
            self.qubo_params["num_bits"],
            fixed_point=self.qubo_params["fixed_point"],
            sign=self.qubo_params["sign"]
        )
        A_discrete = discretize_matrix(self._A, bit_value)
        Q = get_qubo(
            A_discrete, self._b,
            eq_scaling_val=self.qubo_params.get("eq_scaling_val", None)
        )

        # sampling
        sampleset = self.sampler.sample_qubo(Q, **self.sampling_params)

        # recover x from q
        lowest_q = np.fromiter(
            sampleset.first.sample.values(), dtype=np.float64
        )
        x = q2x(lowest_q, bit_value)

        return {"x": x, "sampleset": sampleset}

    @staticmethod
    def default_qubo_params():
        return {
            "num_bits": 4,
            "fixed_point": 1,
            "sign": "pn",
            "eq_scaling_val": 1/8,
        }


class IterativeSolver(BaseSolver):

    required_iter_params = [
        "scale_factor",
    ]

    def __init__(self, A, b,
                 sampler=None,
                 sampling_params=None,
                 qubo_params=None,
                 iter_params=None) -> None:

        self.iter_params = iter_params
        super().__init__(A, b, sampler=sampler,
                         sampling_params=sampling_params,
                         qubo_params=qubo_params)

    def solve(self,
              initial_x,
              sampler=None,
              sampling_params=None,
              qubo_params=None,
              iter_params=None):

        self._assign(sampler, "sampler")
        self._assign(sampling_params, "sampling_params")
        self._assign(
            qubo_params, "qubo_params",
            default_value=self.default_qubo_params(),
            check_func=self._check_qubo_params
        )
        self._assign(
            iter_params, "iter_params",
            default_value=self.default_iter_params(),
            check_func=self._check_iter_params
        )

        # get A_discrete
        bit_value = get_bit_value(
            self.qubo_params["num_bits"],
            fixed_point=self.qubo_params["fixed_point"],
            sign=self.qubo_params["sign"]
        )
        A_discrete = discretize_matrix(self._A, bit_value)

        # parameters for iteration
        ones_vector = np.ones(self._b.size)
        scale_factor = self.iter_params["scale_factor"]
        num_iter = self.iter_params.get("num_iter", 10)
        l2_res_tol = self.iter_params.get("l2_res_tol", 1e-4)
        history = {
            "x": [initial_x],
            "l2_res": [np.linalg.norm(self._A @ initial_x - self._b)]
        }
        if num_iter <= 0:
            num_iter = 10
            warnings.warn("Set num_iter=10")

        # Start iteration
        for i in range(num_iter):
            # construct new RHS vector
            tmp_b = (
                self._b
                + scale_factor * (self._A @ ones_vector)
                - self._A @ initial_x
            ) / scale_factor

            # get qubo
            Q = get_qubo(
                A_discrete, tmp_b,
                eq_scaling_val=self.qubo_params.get("eq_scaling_val", None)
            )

            # sampling
            sampleset = self.sampler.sample_qubo(Q, **self.sampling_params)

            # recover improvement vector from q
            lowest_q = np.fromiter(
                sampleset.first.sample.values(), dtype=np.float64)
            improvement_vector = q2x(lowest_q, bit_value)

            # update initial guess
            improvement_x = (
                scale_factor * (improvement_vector - ones_vector)
            )
            x = initial_x + improvement_x
            l2_res = np.linalg.norm(self._A @ x - self._b)
            history["x"].append(x)
            history["l2_res"].append(l2_res)
            if self.iter_params.get("verbose", False):
                with np.printoptions(precision=4):
                    print(f"Iter: {i+1}\n    x: {x}\n    l2_res: {l2_res:.4e}")

            if l2_res <= l2_res_tol:
                break
            else:
                initial_x = x
                # adjust scale factor
                if np.sum(np.abs(improvement_vector - 1) >= 0.5) \
                        == self._b.size:
                    scale_factor /= 0.5
                elif np.sum(np.abs(improvement_vector - 1) >= 0.5) \
                        > self._b.size//2:
                    scale_factor /= 1
                elif np.sum(np.abs(improvement_vector - 1) <= 0.25) \
                        > self._b.size//2:
                    scale_factor /= 1.5
                else:
                    scale_factor /= 2

        return {
            "x": x,
            "history": history
        }

    def _check_iter_params(self, iter_params):
        if not isinstance(iter_params, dict):
            raise TypeError(
                f"iter_params must be a dict, not {type(iter_params)}"
            )
        for param in self.required_iter_params:
            if param not in iter_params:
                raise KeyError(f"{param} must in `iter_params`")

    @staticmethod
    def default_iter_params():
        return {
            "scale_factor": 2,
            "num_iter": 10,
            "l2_res_tol": 1e-3,
            "verbose": False
        }

    @staticmethod
    def default_qubo_params():
        return {
            "num_bits": 2,
            "fixed_point": 1,
            "sign": "p",
            "eq_scaling_val": 1/8,
        }
