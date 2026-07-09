import torch

from tests.math_audit.helpers import central_difference, cosine_similarity


def test_central_difference_matches_quadratic_gradient():
    value = torch.tensor([1.5, -2.0], dtype=torch.float64)

    actual = central_difference(lambda x: x.square().sum(), value)

    assert torch.allclose(actual, 2.0 * value, atol=1e-6, rtol=1e-6)


def test_cosine_similarity_reports_parallel_and_opposite_vectors():
    vector = torch.tensor([1.0, -2.0], dtype=torch.float64)
    one = torch.tensor(1.0, dtype=torch.float64)

    assert torch.allclose(cosine_similarity(vector, 3.0 * vector), one)
    assert torch.allclose(cosine_similarity(vector, -vector), -one)
