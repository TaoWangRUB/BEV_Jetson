"""4-DoF pose graph: position and yaw per keyframe, roll and pitch from the VIO.

A node's rotation is R = Rz(yaw) @ R_rp, with R_rp the VIO rotation with its own yaw removed, so
optimising yaw never disturbs gravity alignment (VINS-Mono's argument; Basalt's roll-pitch factor in
its strongest form). An edge i->j carries the relative pose T_ij = T_i^-1 T_j as (t_ij in frame i,
dyaw_ij) where dyaw_ij = yaw(R_rp_i @ R_ij), the constant that makes yaw_j = yaw_i + dyaw_ij exact.

Residuals: (R_i^T (p_j - p_i) - t_ij) / sigma_t and wrap(yaw_j - yaw_i - dyaw_ij) / sigma_yaw.
Sequential edges have a covariance that grows with travelled distance (the VIO exposes none);
loop edges carry the verifier's and a Huber weight applied by iteratively reweighted least squares
(cuVSLAM's robustifier). Solved with sparse Levenberg-Marquardt via scipy; the first node is fixed.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .camera import Rz, se3, wrap_angle, yaw_of


@dataclass
class Edge:
    i: int
    j: int
    t_ij: np.ndarray
    dyaw: float
    sigma_t: float
    sigma_yaw: float
    loop: bool
    weight: float = 1.0


class PoseGraph4DoF:
    def __init__(self, cfg):
        self.cfg = cfg
        self.t_ns = []
        self.p = []        # current estimate, world
        self.yaw = []
        self.R_rp = []
        self.p_vio = []
        self.R_vio = []
        self.edges = []

    # -- building ---------------------------------------------------------------------------
    def add_node(self, t_ns, T_vio):
        R, p = T_vio[:3, :3], T_vio[:3, 3].copy()
        y = yaw_of(R)
        self.t_ns.append(int(t_ns))
        self.R_vio.append(R.copy())
        self.p_vio.append(p.copy())
        self.R_rp.append(Rz(-y) @ R)
        k = len(self.p)
        if k == 0:
            self.p.append(p.copy())
            self.yaw.append(y)
        else:
            # place the new node by the VIO delta from the previous (already corrected) node
            i = k - 1
            T_ij = self._rel(self.R_vio[i], self.p_vio[i], R, p)
            self.p.append(self.p[i] + self._R(i) @ T_ij[:3, 3])
            self.yaw.append(wrap_angle(self.yaw[i] + yaw_of(self.R_rp[i] @ T_ij[:3, :3])))
            d = float(np.linalg.norm(T_ij[:3, 3]))
            c = self.cfg
            self.edges.append(Edge(i, k, T_ij[:3, 3].copy(), yaw_of(self.R_rp[i] @ T_ij[:3, :3]),
                                   float(c["seq_sigma_t0_m"]) + float(c["seq_sigma_t_per_m"]) * d,
                                   np.radians(float(c["seq_sigma_yaw0_deg"]) + float(c["seq_sigma_yaw_per_m_deg"]) * d),
                                   False))
        return k

    def add_loop_edge(self, i, j, T_ij, sigma_t, sigma_yaw):
        """T_ij = T_i^-1 T_j (pose of j in i's frame)."""
        e = Edge(i, j, T_ij[:3, 3].copy(), yaw_of(self.R_rp[i] @ T_ij[:3, :3]), float(sigma_t), float(sigma_yaw), True)
        self.edges.append(e)
        return e

    @staticmethod
    def _rel(Ri, pi, Rj, pj):
        return se3(Ri.T @ Rj, Ri.T @ (pj - pi))

    def _R(self, k):
        return Rz(self.yaw[k]) @ self.R_rp[k]

    def pose(self, k):
        return se3(self._R(k), self.p[k])

    def poses(self):
        return [self.pose(k) for k in range(len(self.p))]

    # -- solving ------------------------------------------------------------------------------
    def _pack(self):
        return np.concatenate([np.concatenate([self.p[k], [self.yaw[k]]]) for k in range(1, len(self.p))])

    def _unpack(self, x):
        for k in range(1, len(self.p)):
            self.p[k] = x[4 * (k - 1): 4 * (k - 1) + 3].copy()
            self.yaw[k] = float(x[4 * (k - 1) + 3])

    def _residuals(self, x, edges):
        n = len(self.p)
        p = np.empty((n, 3))
        yaw = np.empty(n)
        p[0], yaw[0] = self.p[0], self.yaw[0]
        xx = x.reshape(-1, 4)
        p[1:], yaw[1:] = xx[:, :3], xx[:, 3]
        out = np.empty(4 * len(edges))
        for e_i, e in enumerate(edges):
            Ri = Rz(yaw[e.i]) @ self.R_rp[e.i]
            w = e.weight
            out[4 * e_i: 4 * e_i + 3] = w * (Ri.T @ (p[e.j] - p[e.i]) - e.t_ij) / e.sigma_t
            out[4 * e_i + 3] = w * wrap_angle(yaw[e.j] - yaw[e.i] - e.dyaw) / e.sigma_yaw
        return out

    def _sparsity(self, edges):
        n = len(self.p)
        S = lil_matrix((4 * len(edges), 4 * (n - 1)), dtype=np.int8)
        for e_i, e in enumerate(edges):
            for k in (e.i, e.j):
                if k > 0:
                    S[4 * e_i: 4 * e_i + 4, 4 * (k - 1): 4 * (k - 1) + 4] = 1
        return S.tocsr()

    def optimize(self, irls_rounds=3):
        """Solve; returns the loop edges' normalised residual norms after the final solve."""
        if len(self.p) < 2 or not self.edges:
            return np.zeros(0)
        edges = self.edges
        huber = float(self.cfg["huber_m"])
        for e in edges:
            e.weight = 1.0
        S = self._sparsity(edges)
        x = self._pack()
        for _ in range(irls_rounds if any(e.loop for e in edges) else 1):
            res = least_squares(self._residuals, x, args=(edges,), jac_sparsity=S, method="trf",
                                x_scale="jac", max_nfev=100, xtol=1e-10, ftol=1e-10)
            x = res.x
            changed = False
            for e in edges:
                if not e.loop:
                    continue
                # unweighted translation residual in metres, for the Huber weight
                r = self._edge_residual(x, e)
                dm = np.linalg.norm(r[:3]) * e.sigma_t
                w = 1.0 if dm <= huber else float(np.sqrt(huber / dm))
                if abs(w - e.weight) > 1e-6:
                    changed = True
                e.weight = w
            if not changed:
                break
        self._unpack(x)
        return np.array([np.linalg.norm(self._edge_residual(x, e)) for e in edges if e.loop])

    def _edge_residual(self, x, e):
        """Normalised (sigma-scaled, unweighted) residual of one edge at solution x."""
        n = len(self.p)
        xx = x.reshape(-1, 4)
        p = np.vstack([self.p[0][None], xx[:, :3]]) if n > 1 else np.array([self.p[0]])
        yaw = np.concatenate([[self.yaw[0]], xx[:, 3]]) if n > 1 else np.array([self.yaw[0]])
        Ri = Rz(yaw[e.i]) @ self.R_rp[e.i]
        return np.concatenate([(Ri.T @ (p[e.j] - p[e.i]) - e.t_ij) / e.sigma_t,
                               [wrap_angle(yaw[e.j] - yaw[e.i] - e.dyaw) / e.sigma_yaw]])

    def loop_edges(self):
        return [e for e in self.edges if e.loop]

    def drop_edge(self, e):
        self.edges.remove(e)

    def reduce_to(self, max_nodes):
        """Node cap (cuVSLAM ReduceKeyframes): merge the oldest node without loop edges into its
        successor by composing the sequential edges. Returns the removed node indices (original)."""
        return []  # kept as an explicit no-op until max_nodes > 0 is needed offline (design D6)
