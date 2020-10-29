from collections import defaultdict

import numpy as np
import pytest
from dwaveutils import bl_lstsq
from numpy.testing import assert_array_equal


def test_discretize_matrix():
    """Test `discretize_matrix` function."""
    A = np.array(
        [[1.1, -3.7, 4.1],
            [2.0, 9.9, -5.0]]
    )
    bit_value = np.array([-1 * 2**0, 2**-1, 2**-2])
    A_discrete = bl_lstsq.discretize_matrix(A, bit_value)
    expected_result = np.array([
        [-1.1, 0.55, 0.275, 3.7, -1.85, -0.925, -4.1, 2.05, 1.025],
        [-2.0, 1.0, 0.5, -9.9, 4.95, 2.475,  5.0, -2.5, -1.25]
    ])
    assert_array_equal(A_discrete, expected_result,
                       err_msg='Wrong discretized')


def test_get_bit_value():
    num_bits = 5
    bit_value1 = bl_lstsq.get_bit_value(num_bits)
    expected_result1 = np.array([-1, 0.5, 0.25, 0.125, 0.0625])
    assert_array_equal(bit_value1, expected_result1,
                       err_msg='Wrong bit value')
    bit_value2 = bl_lstsq.get_bit_value(num_bits, fixed_point=2, sign='p')
    expected_result2 = np.array([2, 1, 0.5, 0.25, 0.125])
    assert_array_equal(bit_value2, expected_result2,
                       err_msg='Wrong bit value')
    bit_value3 = bl_lstsq.get_bit_value(num_bits, fixed_point=1, sign='n')
    expected_result3 = np.array([-1, -0.5, -0.25, -0.125, -0.0625])
    assert_array_equal(bit_value3, expected_result3,
                       err_msg='Wrong bit value')


def test_q2x():
    """Test `q_to_x` function."""
    q = np.array([1, 0, 0, 1, 1, 1])
    bit_value = np.array([-1 * 2**0, 2**-1, 2**-2])
    x = bl_lstsq.q2x(q, bit_value)
    expected_result = np.array([-1, -0.25])
    assert_array_equal(x, expected_result, err_msg='Wrong conversion')


def test_bruteforce():
    """Test `bruteforce` function."""
    A_discrete = np.array([
        [-1, 0.5, 0.25, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, 0.5, 0.25, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, 0.5, 0.25],
    ])
    b = np.array([0.75, 0.5, -0.5])
    bit_value = np.array([-1 * 2**0, 2**-1, 2**-2])
    best_q, best_x, min_norm = bl_lstsq.bruteforce(A_discrete, b, bit_value)
    expected_q = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0])
    expected_x = np.array([0.75, 0.5, -0.5])
    expected_norm = 0
    assert_array_equal(best_q, expected_q, err_msg='Wrong q')
    assert_array_equal(best_x, expected_x, err_msg='Wrong x')
    assert min_norm == expected_norm


def test_get_qubo():
    """Test `get_qubo` function."""
    A_discrete = np.array([
        [-1, 0.5, 0.25, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, 0.5, 0.25, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, 0.5, 0.25],
    ])
    b = np.array([0.75, 0.5, -0.5])
    Q = bl_lstsq.get_qubo(A_discrete, b, eq_scaling_val=1/2)
    expected_Q = defaultdict(
        int, {
            (0, 0): 1.25,
            (1, 1): -0.25,
            (1, 0): -0.5,
            (2, 2): -0.15625,
            (2, 0): -0.25,
            (2, 1): 0.125,
            (3, 3): 1.0,
            (4, 4): -0.125,
            (4, 3): -0.5,
            (5, 5): -0.09375,
            (5, 3): -0.25,
            (5, 4): 0.125,
            (7, 7): 0.375,
            (7, 6): -0.5,
            (8, 8): 0.15625,
            (8, 6): -0.25,
            (8, 7): 0.125
        }
    )
    assert Q == expected_Q, 'Wrong QUBO'


if __name__ == '__main__':
    pytest.main([__file__])
