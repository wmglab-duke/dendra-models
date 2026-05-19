from __future__ import annotations
from typing import Dict, Tuple

import numpy as np
import torch

from dendra.models.parametric import Parametric, Bounded


class PairwiseDistanceStore:
    """
    Cache pairwise distances between named ND point clouds without duplication.

    - Stores exactly one matrix for an unordered pair (A,B), keyed by (min(A,B), max(A,B)).
    - On query, returns the stored matrix or its transpose to match (A,B) order.
    - Uses a low-memory Euclidean computation; optionally block for very large arrays.

    Notes
    -----
    * Arrays are treated as immutable once added; if you mutate/replace an array,
      call `add(name, arr, overwrite=True)` or `clear_distances_for(name)` to avoid stale caches.
    * All distances are standard Euclidean in R^D, where D = pts.shape[1].
    """

    def __init__(self, dtype=np.float64):
        self._points: Dict[str, np.ndarray] = {}
        self._dists: Dict[Tuple[str, str], np.ndarray] = {}
        self._dtype = dtype

    # -------------------- data management --------------------
    def add(self, name: str, pts: np.ndarray, overwrite: bool = False) -> None:
        """
        Register a point set.

        Parameters
        ----------
        name : str
            Unique name for this point cloud.
        pts : (N,D) ndarray
            ND points (D can be any positive integer).
        overwrite : bool
            If True, replace existing and invalidate any cached distances touching `name`.
        """
        pts = np.asarray(pts, dtype=self._dtype)
        if pts.ndim != 2 or pts.shape[1] < 1:
            raise ValueError(f"{name!r} must be a 2D array of shape (N,D) with D>=1; got {pts.shape}")
        if (name in self._points) and not overwrite:
            raise ValueError(f"Points {name!r} already exist. Use overwrite=True to replace.")
        self._points[name] = pts
        # Invalidate distances involving `name` (safe even if none exist yet)
        self.clear_distances_for(name)

    def clear_distances_for(self, name: str) -> None:
        """Drop cached distances that involve `name`."""
        to_drop = [key for key in self._dists.keys() if name in key]
        for k in to_drop:
            self._dists.pop(k, None)

    # -------------------- querying --------------------
    def get(self, name_a: str, name_b: str, block: int | None = None) -> np.ndarray:
        """
        Get the (len(name_a), len(name_b)) Euclidean distance matrix.

        Parameters
        ----------
        name_a, name_b : str
            Names of point sets. If equal, returns square self-distance matrix.
        block : int or None
            If set, compute in blocks of up to `block` rows from A to cap memory.

        Returns
        -------
        D : (NA, NB) ndarray
            Euclidean distances. If the cached key order is (min, max) != (name_a, name_b),
            this returns a view `D.T` to satisfy the requested orientation.
        """
        if name_a not in self._points or name_b not in self._points:
            missing = [n for n in (name_a, name_b) if n not in self._points]
            raise KeyError(f"Unknown point set(s): {missing}")

        # Ensure both point sets have the same ambient dimension
        Da = self._points[name_a].shape[1]
        Db = self._points[name_b].shape[1]
        if Da != Db:
            raise ValueError(f"Dimension mismatch: {name_a} has D={Da}, {name_b} has D={Db}")

        # Canonical unordered key
        key = (name_a, name_b) if name_a <= name_b else (name_b, name_a)

        # Compute and cache if missing
        if key not in self._dists:
            A = self._points[key[0]]
            B = self._points[key[1]]
            self._dists[key] = self._pairwise_euclid(A, B, block=block)

        D = self._dists[key]
        return D if (name_a, name_b) == key else D.T

    # -------------------- core computation --------------------
    @staticmethod
    def _pairwise_euclid(A: np.ndarray,
                         B: np.ndarray,
                         block: int | None = None,
                         *,
                         center_and_scale: bool = True,
                         check_finite: bool = True) -> np.ndarray:
        """
        Compute pairwise Euclidean distances between rows of A (NA,D) and B (NB,D)
        in a numerically robust way.

        - Centers A and B by a common shift to remove large offsets.
        - Optionally rescales to avoid overflow/underflow, then rescales back.
        - Computes in float64 regardless of input dtype (storage still in self._dtype).
        - Supports blocked computation over rows of A.
        """
        A = np.asarray(A)
        B = np.asarray(B)

        if check_finite:
            if not np.isfinite(A).all() or not np.isfinite(B).all():
                badA = np.argwhere(~np.isfinite(A))
                badB = np.argwhere(~np.isfinite(B))
                raise ValueError(
                    f"Non-finite values in inputs: "
                    f"A bad count={badA.shape[0]}, B bad count={badB.shape[0]} "
                    f"(showing up to 5 examples) "
                    f"A{badA[:5].tolist()} B{badB[:5].tolist()}"
                )

        # Work in float64 for stability
        A64 = A.astype(np.float64, copy=False)
        B64 = B.astype(np.float64, copy=False)

        # Common centering and optional scaling (translation/scale handling)
        if center_and_scale:
            shift = (A64.mean(axis=0, dtype=np.float64) + B64.mean(axis=0, dtype=np.float64)) / 2.0
            A0 = A64 - shift
            B0 = B64 - shift
            maxabs = max(np.max(np.abs(A0)), np.max(np.abs(B0)))
            if np.isfinite(maxabs) and maxabs > 0.0:
                scale = maxabs
                inv = 1.0 / scale
                A0 *= inv
                B0 *= inv
            else:
                scale = 1.0
        else:
            A0, B0, scale = A64, B64, 1.0

        if block is None:
            # Gram trick in 64-bit, after centering/scaling
            AA = np.einsum('ij,ij->i', A0, A0)[:, None]   # (NA,1)
            BB = np.einsum('ij,ij->i', B0, B0)[None, :]   # (1,NB)
            G  = A0 @ B0.T                                # (NA,NB)
            D2 = AA + BB - 2.0 * G
            np.maximum(D2, 0.0, out=D2)
            return np.sqrt(D2, out=D2) * scale

        # ---- Blocked over rows of A ----
        NA, NB = A0.shape[0], B0.shape[0]
        out = np.empty((NA, NB), dtype=np.float64)
        BB = np.einsum('ij,ij->i', B0, B0)               # (NB,)
        BT = B0.T                                        # (D,NB)

        for i0 in range(0, NA, block):
            i1 = min(i0 + block, NA)
            Ai = A0[i0:i1]                               # (bi,D)
            AA = np.einsum('ij,ij->i', Ai, Ai)[:, None]  # (bi,1)
            # Gram trick for the block
            D2 = AA + BB[None, :] - 2.0 * (Ai @ BT)      # (bi,NB)
            # If something still goes bad (extreme data), fall back locally
            if not np.isfinite(D2).all():
                # Safer (but heavier) per-block difference formula
                # Compute in sub-tiles to control memory
                bj = 1 + (1 << 12) // max(1, Ai.shape[1])  # crude sizing
                for j0 in range(0, NB, bj):
                    j1 = min(j0 + bj, NB)
                    Bj = B0[j0:j1]                         # (bj,D)
                    diff = Ai[:, None, :] - Bj[None, :, :]
                    Dij2 = np.einsum('ijk,ijk->ij', diff, diff)
                    np.maximum(Dij2, 0.0, out=Dij2)
                    out[i0:i1, j0:j1] = np.sqrt(Dij2, out=Dij2)
            else:
                np.maximum(D2, 0.0, out=D2)
                out[i0:i1] = np.sqrt(D2, out=D2)

        return out * scale
    

class Delay(Parametric):
    
    def __init__(self, dist, cv, syn_delay):
        super().__init__()
        self.n = len(dist)
        self.syn_delay = syn_delay
        cv = torch.as_tensor(cv).expand(self.n)
        self.cv = Bounded(cv, min_val=300.0, max_val=800.0)
        self.register_buffer('dist', torch.as_tensor(dist))

    def __len__(self):
        return self.n

    def forward(self):
        cv = self.cv(cache=(not self.training))
        return self.syn_delay + self.dist / cv

    def repeat(self, n):
        return self().repeat(n)
    
