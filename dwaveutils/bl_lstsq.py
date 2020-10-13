"""binary linear least square."""

from collections import defaultdict

import numpy as np
from tqdm import tqdm


def discretize_matrix(matrix, bit_value):
    return np.kron(matrix, bit_value)


def get_bit_value(num_bits, fixed_point=0):
    """The value of each bit in two's-complement binary fixed-point numbers."""
    return np.array([-2 ** fixed_point if i == 0
                     else 2. ** (fixed_point - i)
                     for i in range(0, num_bits)])


def setup_nbit_laplacian(N, num_bits,
                         fixed_point=0, exact_x=True, random_seed=None):
    """Get information about 1-D laplace equation."""

    # number of predictor and number of response
    num_predictor_discrete = num_bits * N
    num_response = N

    # matrix `A`
    A = (np.eye(num_response, k=-1)
         - 2 * np.eye(num_response, k=0)
         + np.eye(num_response, k=1))
    # set the bit value to discrete the actual value as a fixed point
    bit_value = get_bit_value(num_bits, fixed_point=fixed_point)
    # discretized version of matrix `A`
    A_discrete = discretize_matrix(A, bit_value)

    if random_seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(random_seed)

    if exact_x:
        # binary vector `q`
        q = rng.choice([0, 1], size=num_predictor_discrete)
        # vector `x`
        x = q2x(q, bit_value)
    else:
        # vector `x`
        x = (rng.choice([-1, 1], size=num_response)
             * (2 ** fixed_point) * rng.random(num_response))

    # calculate vector `b`
    b = A @ x

    output = {
        'A': A,
        'x': x,
        'b': b,
        'A_discrete': A_discrete,
        'bit_value': bit_value
    }
    return output


def bruteforce(A_discrete, b):
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

    # number of predictor and number of response
    num_predictor = A_discrete.shape[1]
    num_response = b.size  # or A_discrete.shape[0]

    # initialize QUBO
    qubo_a = np.zeros(num_predictor)
    qubo_b = np.zeros((num_predictor, num_predictor))
    Q = defaultdict(int)

    # define weights
    for i in range(num_response):
        for j in range(num_predictor):
            qubo_a[j] += A_discrete[i, j] * (A_discrete[i, j] - 2 * b[i])
            for k in range(j):
                qubo_b[j, k] += 2 * A_discrete[i, j] * A_discrete[i, k]

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
    import warnings

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from neal import SimulatedAnnealingSampler
    from dwave.system import EmbeddingComposite, DWaveSampler

    # setting
    # size of symmetric matrix `A`
    N = 3
    # number of bits (include sign bit)
    num_bits = 4
    # n-vector bit value is defined by
    # [-2**(fixed_point), 2**(fixed_point-1), ..., 2**(fixed_point-n)]
    fixed_point = 0
    # whether x can be perfectly discrete
    exact_x = False
    random_seed = 19937
    # scaling factor for QUBO
    eq_scaling_val = 1/8
    # number of reads for Simulated annealing (SA) or Quantum annealing (QA)
    num_reads = 1000
    # sampler type must be one of {'SA', 'QA'}
    sampler_type = 'SA'

    # setup A, x, b, A_discrete, bit_value
    output = setup_nbit_laplacian(
        N, num_bits,
        fixed_point=fixed_point, exact_x=exact_x, random_seed=random_seed
    )
    A = output['A']
    true_x = output['x']
    true_b = output['b']
    A_discrete = output['A_discrete']
    bit_value = output['bit_value']

    # solve A_discrete*q=b where q is a binary vector by simulated annealing
    Q = get_qubo(A_discrete, true_b, eq_scaling_val=eq_scaling_val)
    if sampler_type == 'QA':
        try:
            sampler = EmbeddingComposite(DWaveSampler(solver={'qpu': True}))
            _sampler_args = {}
            if 'num_reads' in sampler.parameters:
                _sampler_args['num_reads'] = num_reads
            if 'answer_mode' in sampler.parameters:
                _sampler_args['answer_mode'] = 'raw'
            sampleset = sampler.sample_qubo(Q, **_sampler_args)
        except ValueError:
            warnings.warn('Cannot access QPU, use \
                           SimulatedAnnealingSampler instead.')
            sampler = SimulatedAnnealingSampler()
            sampleset = sampler.sample_qubo(Q, num_reads=num_reads)
    elif sampler_type == 'SA':
        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample_qubo(Q, num_reads=num_reads)
    else:
        raise(ValueError("The sampler_type is wrong, \
                          please enter 'SA' or 'QA'"))

    # solve A_discrete*q=b where q is a binary vector by brute force
    # Warning: this may take a lot of time!
    best_q, best_x, min_norm = bruteforce(A_discrete, true_b)

    # prepare for showing results and plotting
    # convert sampleset and its aggregate version to dataframe
    sampleset_pd = sampleset.to_pandas_dataframe()
    sampleset_pd_agg = sampleset.aggregate().to_pandas_dataframe()
    num_states = len(sampleset_pd_agg)
    num_b_entry = len(true_b)
    num_x_entry = len(true_x)
    num_q_entry = A_discrete.shape[1]
    # concatnate `sampleset_pd` and `x_at_each_read`
    x_at_each_read = pd.DataFrame(
        np.row_stack(
            [(sampleset_pd.iloc[i][:num_q_entry]).values.reshape(
                (num_x_entry, -1)) @ bit_value
             for i in range(num_reads)]
        ),
        columns=['x' + str(i) for i in range(num_x_entry)]
    )
    sampleset_pd = pd.concat([sampleset_pd, x_at_each_read], axis=1)
    sampleset_pd.rename(
        columns=lambda c: c if isinstance(c, str) else 'q'+str(c),
        inplace=True
    )
    # concatnate `sampleset_pd_agg` and `x_at_each_state`
    x_at_each_state = pd.DataFrame(
        np.row_stack(
            [(sampleset_pd_agg.iloc[i][:num_q_entry]).values.reshape(
                (num_x_entry, -1)) @ bit_value
             for i in range(num_states)]
        ),
        columns=['x' + str(i) for i in range(num_x_entry)]
    )
    sampleset_pd_agg = pd.concat([sampleset_pd_agg, x_at_each_state], axis=1)
    sampleset_pd_agg.rename(
        columns=lambda c: c if isinstance(c, str) else 'q'+str(c),
        inplace=True
    )
    # lowest energy state x and q
    lowest_q = sampleset_pd_agg.sort_values(
        'energy').iloc[0, :num_q_entry].values
    lowest_x = q2x(lowest_q, bit_value)
    # frequently occurring x and q
    frequent_q = sampleset_pd_agg.sort_values(
        'num_occurrences', ascending=False).iloc[0, :num_q_entry].values
    frequent_x = q2x(frequent_q, bit_value)
    # calculate expected x from x
    expected_x = sampleset_pd_agg.apply(
        lambda row: row.iloc[-num_x_entry:]
        * (row.num_occurrences / num_reads),
        axis=1
    ).sum().values
    # calculate excepted x from q
    tmp_q = sampleset_pd_agg.apply(
        lambda row: row.iloc[:num_q_entry]
        * (row.num_occurrences / num_reads),
        axis=1
    ).sum() > 0.5  # bool
    expected_x_discrete = q2x(tmp_q, bit_value)

    # show results
    print('='*50)
    print('true x:', true_x)
    print('true b:', true_b)
    print('bit value:', bit_value)
    print('='*50)
    print('# brute force')
    print('best x:', best_x)
    print('best q:', best_q)
    print('b:', A @ best_x)
    print('2-norm:', min_norm)
    print('='*50)
    print('# Simulated annealing/Quantum annealing')
    print('lowest energy state x:')
    print(lowest_x)
    print('lowest energy state q:')
    print(lowest_q)
    print('b:', A @ lowest_x)
    print('2-norm:', np.linalg.norm(A @ lowest_x - true_b))
    print('-'*50)
    print('most frequently occurring x:')
    print(frequent_x)
    print('most frequently occurring q:')
    print(frequent_q)
    print('b:', A @ frequent_x)
    print('2-norm:', np.linalg.norm(A @ frequent_x - true_b))
    print('-'*50)
    print('expected x (from real value):')
    print(expected_x)
    print('b:', A @ expected_x)
    print('2-norm:', np.linalg.norm(A @ expected_x - true_b))
    print('-'*50)
    print('expected x (from discrete value):')
    print(expected_x_discrete)
    print('b:', A @ expected_x_discrete)
    print('2-norm:', np.linalg.norm(A @ expected_x_discrete - true_b))
    print('-'*50)
    print('Sample set:')
    print(sampleset_pd_agg.sort_values('num_occurrences', ascending=False))
    print('='*50)

    # plot histogram
    axes = sampleset_pd.hist(
        figsize=(8, 6), bins=30,
        column=['x' + str(i) for i in range(num_x_entry)],
    )
    axes = axes.ravel()
    for i in range(num_x_entry):
        ax = axes[i]
        ax.set_ylabel('counts')
    plt.tight_layout()
    plt.show()
