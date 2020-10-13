# import unittest
import pytest

import numpy as np
from numpy.testing import assert_array_equal
from dwaveutils import bl_lstsq



def test_discretize_matrix():
    """Test `discretize_matrix` function."""
    A = np.array(
        [[1.1, -3.7, 4.1],
            [2.0, 9.9, -5.0]]
    )
    bit_value = np.array([-1 * 2**0, 2**-1, 2**-2])
    A_discretized = bl_lstsq.discretize_matrix(A, bit_value)
    expected_result = np.array([
        [-1.1, 0.55, 0.275, 3.7, -1.85, -0.925, -4.1, 2.05, 1.025],
        [-2.0, 1.0, 0.5, -9.9, 4.95, 2.475,  5.0, -2.5, -1.25]
    ])
    assert_array_equal(A_discretized, expected_result,
                       err_msg='Wrong discretized')


def test_get_bit_value():
    num_bits = 5
    bit_value = bl_lstsq.get_bit_value(num_bits)
    expected_result = np.array([-1, 0.5, 0.25, 0.125, 0.0625])
    assert_array_equal(bit_value, expected_result,
                       err_msg='Wrong bit value')


def test_q2x():
    """Test `q_to_x` function."""
    q = np.array([1, 0, 0, 1, 1, 1])
    bit_value = np.array([-1 * 2**0, 2**-1, 2**-2])
    x = bl_lstsq.q2x(q, bit_value)
    expected_result = np.array([-1, -0.25])
    assert_array_equal(x, expected_result, err_msg='Wrong conversion')


if __name__ == '__main__':
    pytest.main([__file__])
