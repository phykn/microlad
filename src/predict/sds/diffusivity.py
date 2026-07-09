from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.predict.sds.phase import soft_phase_probability
from src.predict.sds.targets import phase_vector_target
from src.tensors.validation import validate_finite_tensor


LOW_COND_FLOOR = 0.001


class DiffusivitySolver(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        low_cond: float = 0.001,
    ) -> None:
        super().__init__()

        if height < 2:
            raise ValueError("height must be at least 2.")

        if width < 2:
            raise ValueError("width must be at least 2.")

        if low_cond < 0.0 or low_cond > 1.0:
            raise ValueError("low_cond must be between 0 and 1.")

        if low_cond == 0.0:
            low_cond = LOW_COND_FLOOR

        self.height = int(height)
        self.width = int(width)
        self.low_cond = float(low_cond)
        self.nx = self.width + 1
        self.ny = self.height + 1
        self.nn = self.nx * self.ny

        self._build_mesh_buffers()
        unit_response = self._solve_raw(torch.ones(self.height, self.width)).detach()
        self.register_buffer("unit_response", unit_response)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        expected_shape = torch.Size([self.height, self.width])

        if mask.shape != expected_shape:
            raise ValueError(f"mask must have shape [{self.height}, {self.width}].")

        validate_finite_tensor("mask", mask)
        _validate_unit_interval("mask", mask)

        return self._solve_raw(mask) / self.unit_response.to(
            device=self.base_data.device,
            dtype=self.base_data.dtype,
        )

    def _solve_raw(self, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=self.base_data.device, dtype=self.base_data.dtype)
        sigma_e = self.low_cond + (1.0 - self.low_cond) * mask.reshape(
            -1
        ).repeat_interleave(2)
        data = self.base_data * sigma_e[self.elem_idx]

        stiffness = torch.zeros(
            (self.nn, self.nn),
            device=data.device,
            dtype=data.dtype,
        )
        stiffness.index_put_((self.rows, self.cols), data, accumulate=True)

        bc = self.bc_idx
        fc = self.fc_idx

        k_ff = stiffness[fc][:, fc]
        k_fc = stiffness[fc][:, bc]
        u_c = self.u_c.to(dtype=data.dtype)
        b_f = -(k_fc @ u_c)
        u_f = torch.linalg.solve(k_ff, b_f)

        u = torch.zeros(self.nn, device=data.device, dtype=data.dtype)
        u[fc] = u_f
        u[bc] = u_c

        u_e = u[self.elems]
        grad_u = torch.matmul(self.elem_grad_t, u_e.unsqueeze(-1)).squeeze(-1)
        energy = grad_u.pow(2).sum(dim=1)

        return (sigma_e * energy * self.elem_area).sum()

    def _build_mesh_buffers(self) -> None:
        d_hat = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
        elems = []
        for row in range(self.height):
            for col in range(self.width):
                n0 = row * self.nx + col
                n1 = n0 + 1
                n2 = n0 + self.nx
                n3 = n2 + 1
                elems.append((n0, n1, n2))
                elems.append((n1, n3, n2))

        elems_array = np.array(elems, dtype=np.int64)
        rows: list[int] = []
        cols: list[int] = []
        base_data: list[float] = []
        elem_idx: list[int] = []
        grad_t_list = []
        area_list = []

        for index, elem in enumerate(elems_array):
            points = np.array(
                [[elem[node] % self.nx, elem[node] // self.nx] for node in range(3)],
                dtype=float,
            )
            jacobian = np.vstack((points[1] - points[0], points[2] - points[0])).T
            det_j = np.linalg.det(jacobian)
            inv_j_t = np.linalg.inv(jacobian).T
            grads = d_hat.dot(inv_j_t)
            grad_t = grads.T
            area = abs(det_j) / 2.0
            ke0 = area * grads.dot(grads.T)

            for a in range(3):
                for b in range(3):
                    rows.append(int(elem[a]))
                    cols.append(int(elem[b]))
                    base_data.append(float(ke0[a, b]))
                    elem_idx.append(index)

            grad_t_list.append(grad_t)
            area_list.append(area)

        self.register_buffer("elems", torch.from_numpy(elems_array).long())
        self.register_buffer("rows", torch.tensor(rows, dtype=torch.long))
        self.register_buffer("cols", torch.tensor(cols, dtype=torch.long))
        self.register_buffer("base_data", torch.tensor(base_data, dtype=torch.float32))
        self.register_buffer("elem_idx", torch.tensor(elem_idx, dtype=torch.long))
        self.register_buffer(
            "elem_grad_t",
            torch.tensor(np.stack(grad_t_list), dtype=torch.float32),
        )
        self.register_buffer("elem_area", torch.tensor(area_list, dtype=torch.float32))

        bc = []

        for row in range(self.ny):
            bc.append(row * self.nx)
            bc.append(row * self.nx + self.nx - 1)

        bc_idx = np.unique(bc)
        fc_idx = np.setdiff1d(np.arange(self.nn), bc_idx)

        u_c = np.zeros(len(bc_idx), dtype=np.float32)
        coords = np.stack([bc_idx % self.nx, bc_idx // self.nx], axis=1)
        u_c[coords[:, 0] == 0] = 1.0

        self.register_buffer("bc_idx", torch.tensor(bc_idx, dtype=torch.long))
        self.register_buffer("fc_idx", torch.tensor(fc_idx, dtype=torch.long))
        self.register_buffer("u_c", torch.tensor(u_c, dtype=torch.float32))


def diffusivity_loss(
    values: torch.Tensor,
    targets: Mapping[int, float] | torch.Tensor,
    *,
    solver: DiffusivitySolver,
    num_phases: int,
    temperature: float = 0.1,
    weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if values.ndim < 2:
        raise ValueError("values must have at least two spatial dimensions.")

    if values.numel() == 0 or values.shape[-2] <= 0 or values.shape[-1] <= 0:
        raise ValueError("values must have non-empty spatial dimensions.")

    if values.ndim == 4 and values.shape[1] != 1:
        raise ValueError("values with 4 dimensions must have shape [B, 1, H, W].")

    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")

    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")

    if weight < 0.0:
        raise ValueError("weight must be non-negative.")

    actual_diffusivity = compute_diffusivity(
        values,
        solver=solver,
        num_phases=num_phases,
        temperature=temperature,
    )
    target_diffusivity = phase_vector_target(
        targets,
        num_phases=num_phases,
        device=actual_diffusivity.device,
        dtype=actual_diffusivity.dtype,
        label="diffusivity value",
    )
    validate_finite_tensor("diffusivity targets", target_diffusivity)
    target_diffusivity = target_diffusivity.clamp(min=solver.low_cond, max=1.0)

    loss = weight * F.mse_loss(actual_diffusivity, target_diffusivity)

    stats = {
        "actual_diffusivity": actual_diffusivity.detach(),
        "target_diffusivity": target_diffusivity.detach(),
    }

    return loss, stats


def compute_diffusivity(
    values: torch.Tensor,
    *,
    solver: DiffusivitySolver,
    num_phases: int,
    temperature: float = 0.1,
) -> torch.Tensor:
    if values.ndim < 2:
        raise ValueError("values must have at least two spatial dimensions.")

    if values.numel() == 0 or values.shape[-2] <= 0 or values.shape[-1] <= 0:
        raise ValueError("values must have non-empty spatial dimensions.")

    if values.ndim == 4 and values.shape[1] != 1:
        raise ValueError("values with 4 dimensions must have shape [B, 1, H, W].")

    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")

    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")

    validate_finite_tensor("values", values)

    height, width = values.shape[-2:]
    slices = values.reshape(-1, height, width)
    probability = soft_phase_probability(
        slices,
        num_phases=num_phases,
        temperature=temperature,
        phase_dim=1,
    )
    if probability.shape[-2:] != (solver.height, solver.width):
        probability = F.interpolate(
            probability,
            size=(solver.height, solver.width),
            mode="bilinear",
            align_corners=False,
        )

    actual = []

    for phase in range(num_phases):
        phase_values = []

        for slice_index in range(probability.shape[0]):
            phase_values.append(solver(probability[slice_index, phase]))

        actual.append(torch.stack(phase_values).mean())

    return torch.stack(actual)


def _validate_unit_interval(name: str, values: torch.Tensor) -> None:
    if values.min().item() < 0.0 or values.max().item() > 1.0:
        raise ValueError(f"{name} values must be between 0 and 1.")
