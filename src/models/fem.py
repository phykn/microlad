import numpy as np
import torch
import torch.nn as nn


class TorchFEMMesh(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        low_cond: float = 0.001,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.nx = width + 1
        self.ny = height + 1
        self.nn = self.nx * self.ny
        self.low_cond = low_cond
        self.device = torch.device(device)

        d_hat = np.array([[-1, -1], [1, 0], [0, 1]], dtype=float)
        elems = []
        for j in range(height):
            for i in range(width):
                n0 = j * self.nx + i
                n1 = n0 + 1
                n2 = n0 + self.nx
                n3 = n2 + 1
                elems.append((n0, n1, n2))
                elems.append((n1, n3, n2))

        elems = np.array(elems, dtype=np.int64)
        self.register_buffer("elems", torch.from_numpy(elems))

        rows, cols, base_data = [], [], []
        elem_idx = []
        grad_t_list = []
        area_list = []
        for idx, elem in enumerate(elems):
            pts = np.array([[elem[k] % self.nx, elem[k] // self.nx] for k in range(3)], float)
            jacobian = np.vstack((pts[1] - pts[0], pts[2] - pts[0])).T
            det_j = np.linalg.det(jacobian)
            inv_j_t = np.linalg.inv(jacobian).T
            grads = d_hat.dot(inv_j_t)
            grad_t = grads.T
            area = abs(det_j) / 2.0
            ke0 = area * grads.dot(grads.T)

            for a in range(3):
                for b in range(3):
                    rows.append(elem[a])
                    cols.append(elem[b])
                    base_data.append(ke0[a, b])
                    elem_idx.append(idx)

            grad_t_list.append(grad_t)
            area_list.append(area)

        self.register_buffer("rows", torch.LongTensor(rows))
        self.register_buffer("cols", torch.LongTensor(cols))
        self.register_buffer("base_data", torch.tensor(base_data, dtype=torch.float32))
        self.register_buffer("elem_idx", torch.LongTensor(elem_idx))
        self.register_buffer("elem_grad_t", torch.tensor(np.stack(grad_t_list), dtype=torch.float32))
        self.register_buffer("elem_area", torch.tensor(area_list, dtype=torch.float32))

        bc = []
        for j in range(self.ny):
            bc.append(j * self.nx)
            bc.append(j * self.nx + (self.nx - 1))

        bc_idx = np.unique(bc)
        fc_idx = np.setdiff1d(np.arange(self.nn), bc_idx)
        self.register_buffer("bc_idx", torch.LongTensor(bc_idx))
        self.register_buffer("fc_idx", torch.LongTensor(fc_idx))

        u_c = np.zeros(len(bc_idx), dtype=np.float32)
        coords = np.stack([bc_idx % self.nx, bc_idx // self.nx], -1)
        u_c[coords[:, 0] == 0] = 1.0
        u_c[coords[:, 0] == self.nx - 1] = 0.0
        self.register_buffer("u_c", torch.tensor(u_c))

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.shape != torch.Size([self.height, self.width]):
            raise ValueError(f"mask must have shape [{self.height}, {self.width}].")

        mask = mask.to(self.base_data.device)
        sigma_e = self.low_cond + (1.0 - self.low_cond) * mask.reshape(-1).repeat_interleave(2)
        data = self.base_data * sigma_e[self.elem_idx]
        stiffness = torch.zeros((self.nn, self.nn), device=self.base_data.device, dtype=data.dtype)
        stiffness.index_put_((self.rows, self.cols), data, accumulate=True)

        bc, fc = self.bc_idx, self.fc_idx
        k_ff = stiffness[fc][:, fc]
        k_fc = stiffness[fc][:, bc]
        b_f = -(k_fc @ self.u_c)
        u_f = torch.linalg.solve(k_ff, b_f)

        u = torch.zeros(self.nn, device=self.base_data.device)
        u[fc] = u_f
        u[bc] = self.u_c

        u_e = u[self.elems]
        grad_u = torch.matmul(self.elem_grad_t, u_e.unsqueeze(-1)).squeeze(-1)
        sq = (grad_u**2).sum(dim=1)
        return (sigma_e * sq * self.elem_area).sum()
