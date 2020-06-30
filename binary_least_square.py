import numpy as np
from collections import defaultdict


def setup_nbit_laplacian(N, n, seed=None):
    '''Get matrix `A` and vector `b`.'''

    # number of predictor and number of response
    num_predictor = n * N
    num_response = N

    # initialize matrix `A`
    A = np.zeros((num_response, num_predictor), dtype=np.float)
    # set the bit value to discrete the actual value as a fixed point
    bit_value = np.array([2. ** -i for i in range(1, n + 1)])
    # assign values to matrix `A`
    # boundary
    for i in range(0, n):
        A[0, i] = -2 * bit_value[i]
        A[0, n + i] = 1 * bit_value[i]
        # A[N - 1, -2 * n + i]
        A[-1, -2 * n + i] = 1 * bit_value[i]
        # A[N - 1, -1 - (n - 1) + i]
        A[-1, -1 - (n - 1) + i] = -2 * bit_value[i]
    # general
    for i in range(1, num_response - 1):
        for j in range(0, n):
            A[i, n * (i + 1) - 2 * n + (j + 1) - 1] = 1 * bit_value[j]
            A[i, n * (i + 1) - n + (j + 1) - 1] = -2 * bit_value[j]
            A[i, n * (i + 1) + (j + 1) - 1] = 1 * bit_value[j]

    # initialize vector `x` and assign values to vector `x`
    # x = np.zeros(num_predictor)
    # for i in range(0, num_response):
    #     x[i * n] = 1
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)
    x = np.array(
        [
            int(x)
            for x in format(
                rng.integers(0, 2 ** (num_predictor)),
                f'0{num_predictor}b'
            )
        ]
    )

    # calculate vector `b`
    b = A @ x

    return A, b


def bruteforce(A, b):
    '''Solve A*x=b where x is a binary vector by brute force.'''

    # number of predictor
    num_predictor = A.shape[1]
    # initialize minimum 2-norm
    min_norm = np.inf

    # loop all solution
    for i in range(0, 2 ** num_predictor):
        # assign solution
        # https://stackoverflow.com/questions/13557937/how-to-convert-decimal-to-binary-list-in-python/13558001
        # https://stackoverflow.com/questions/13522773/convert-an-integer-to-binary-without-using-the-built-in-bin-function
        x = np.array([int(bit) for bit in format(i, f'0{num_predictor}b')])
        # calculate 2-norm
        new_norm = np.linalg.norm(A @ x - b, 2)
        # update best solution
        if new_norm < min_norm:
            min_norm = new_norm
            best_x = np.copy(x)

    return best_x, min_norm


def get_qubo(A, b, eq_scaling_val=1 / 8):
    '''
    Get coefficients of a quadratic unconstrained binary optimization (QUBO)
    problem defined by the dictionary.
    '''

    # number of predictor and number of response
    num_predictor = A.shape[1]
    num_response = b.size  # or A.shape[0]

    # initialize QUBO
    qubo_a = np.zeros(num_predictor)
    qubo_b = np.zeros((num_predictor, num_predictor))
    Q = defaultdict(int)

    # define weights
    for i in range(num_response):
        for j in range(num_predictor):
            qubo_a[j] += A[i, j] * (A[i, j] - 2 * b[i])
            for k in range(j):
                qubo_b[j, k] += 2 * A[i, j] * A[i, k]

    # define objective
    for i in range(num_predictor):
        if qubo_a[i] != 0:
            Q[(i, i)] = eq_scaling_val * qubo_a[i]
        for j in range(num_predictor):
            if qubo_b[i, j] != 0:
                Q[(i, j)] = eq_scaling_val * qubo_b[i, j]

    # define constrait
    # for i in range(qubo_a.size):
    #     Q[(i, i)] += 0.0001

    return Q


if __name__ == "__main__":
    import neal
    # from dwave.system import EmbeddingComposite, DWaveSampler

    # initialize A, b
    N = 3
    n = 4
    seed = 19937
    A, b = setup_nbit_laplacian(N, n, seed=seed)

    # solve A*x=b where x is a binary vector by simulated annealing
    Q = get_qubo(A, b, eq_scaling_val=1 / 8)
    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(Q, num_reads=1000)

    # solve A*x=b where x is a binary vector by brute force
    best_x, min_norm = bruteforce(A, b)

    # show results
    print('best x:', best_x, ', minimum 2-norm:', min_norm)
    print(
        'Sample set:\n',
        sampleset.aggregate()
                 .to_pandas_dataframe()
                 .sort_values('num_occurrences', ascending=False),
        sep=''
    )
