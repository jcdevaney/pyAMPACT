import unittest
import numpy as np
from scipy.linalg import cholesky
from scipy.stats import multivariate_normal

import sys
import os
sys.path.append(os.pardir)
from audioHelpers import mixgauss_prob

class TestMixGaussProb(unittest.TestCase):
    def test_mixgauss_prob_shape(self):
        # Test that the output shape matches the expected shape
        data = np.random.rand(3, 451)
        means = np.random.rand(3, 5)
        covariances = [np.eye(5), np.eye(5)]
        weights = [0.5, 0.5]

        likelihood_matrix = mixgauss_prob(data, means, covariances, weights)
        expected_shape = (3, len(weights))
        self.assertEqual(likelihood_matrix.shape, expected_shape)

    def test_mixgauss_prob_values(self):
        # Test the correctness of the likelihood values
        data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        means = np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]])
        covariances = [np.eye(5), np.eye(5)]
        weights = [0.5, 0.5]

        likelihood_matrix = mixgauss_prob(data, means, covariances, weights)

        # Calculate expected likelihood values using multivariate_normal.pdf
        expected_likelihood = np.array([
            weights[0] * multivariate_normal.pdf(data, mean=means[0], cov=covariances[0]),
            weights[1] * multivariate_normal.pdf(data, mean=means[1], cov=covariances[1])
        ])
        np.testing.assert_allclose(likelihood_matrix, expected_likelihood)

if __name__ == '__main__':
    unittest.main()