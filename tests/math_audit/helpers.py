import torch


def central_difference(
    function,
    value: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    result = torch.empty_like(value)
    flat_value = value.reshape(-1)
    flat_result = result.reshape(-1)

    for index in range(flat_value.numel()):
        delta = torch.zeros_like(flat_value)
        delta[index] = epsilon
        plus = function((flat_value + delta).reshape_as(value))
        minus = function((flat_value - delta).reshape_as(value))
        flat_result[index] = (plus - minus) / (2.0 * epsilon)

    return result


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.reshape(-1)
    right = right.reshape(-1)
    return torch.dot(left, right) / (
        torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    )
