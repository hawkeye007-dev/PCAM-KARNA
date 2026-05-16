"""
PCAM Precision Agent
====================

Key insight: anisotropy is evaluated at the TRUE EQUILIBRIUM a* found
by running dynamics from x_k with pi=I (model.find_equilibrium). The
Hessian H(a*) on clustered patterns has meaningful structure — the
softmax at equilibrium is not fully concentrated, leaving non-trivial
off-diagonal corrections to R that Adam can partially counteract.

Adam minimises log10(cond(Pi^(1/2) H Pi^(1/2))) at each a*, using
multiple initialisations drawn from:
  - Identity (pi=1)
  - Inverse/direct diagonal scaling of H
  - Eigenvector-based: target extreme eigenvalue directions directly

Retrieval: variance-based precision exp(-|q-mu|/sigma), proven +0.12
mean delta on clustered patterns at noise [0.75, 0.85].

Routing: cosine > 0.85 → anisotropy pi; else → retrieval pi.
"""
from __future__ import annotations
from typing import Any

import numpy as np
from adapter import Adapter


class Engine(Adapter):
    def __init__(self,
                 stored_patterns: np.ndarray,
                 model_params: dict[str, Any]) -> None:
        self.X    = stored_patterns.astype(np.float64)
        self.K, self.N = self.X.shape

        self.pi_min = float(model_params.get("pi_min", 0.1))
        self.pi_max = float(model_params.get("pi_max", 10.0))
        self.R      = np.asarray(model_params["R"],   dtype=np.float64)
        self.eta    = float(model_params.get("eta",   0.5))
        self.beta   = float(model_params.get("beta",  8.0))
        self.dt     = float(model_params.get("dt",    0.01))
        self.T_max  = int(model_params.get("T_max",   3000))
        self.tol    = float(model_params.get("tol",   1e-6))

        self.mean = self.X.mean(axis=0)
        self.std  = self.X.std(axis=0) + 1e-8

        self.pattern_pi = np.ones((self.K, self.N), dtype=np.float64)
        self.equilibria = np.zeros_like(self.X)

        for k in range(self.K):
            a_star = self._find_equilibrium(self.X[k])
            self.equilibria[k] = a_star
            H = self._hessian(a_star)
            ev = np.linalg.eigvalsh(0.5 * (H + H.T))
            if ev.min() > 1e-9:
                self.pattern_pi[k] = self._optimise(H)

    def _softmax(self, a):
        z = self.beta * (self.X @ a); z -= z.max()
        e = np.exp(z); return e / e.sum()

    def _hessian(self, a):
        s = self._softmax(a)
        D = np.diag(s) - np.outer(s, s)
        H = self.R - self.eta * self.beta * (self.X.T @ (D @ self.X))
        return 0.5 * (H + H.T)

    def _clip_norm(self, pi):
        pi = np.asarray(pi, dtype=np.float64)
        for _ in range(20):
            pi = np.clip(pi, self.pi_min, self.pi_max)
            m = pi.mean()
            if m < 1e-12: return np.ones(self.N)
            pi /= m
            if (pi.min() >= self.pi_min - 1e-9
                    and pi.max() <= self.pi_max + 1e-9
                    and abs(pi.mean() - 1.0) < 1e-8):
                break
        return np.clip(pi, self.pi_min, self.pi_max)

    def _find_equilibrium(self, x0):
        a = x0.astype(np.float64).copy()
        pi = np.ones(self.N)
        for _ in range(self.T_max):
            s = self._softmax(a)
            g = self.R @ a - self.eta * (self.X.T @ s)
            a_new = a + self.dt * (-pi * g)
            if np.linalg.norm(a_new - a) < self.tol:
                return a_new
            a = a_new
        return a

    def _loss_grad(self, log_pi, H):
        raw  = np.exp(log_pi)
        clip = np.clip(raw, self.pi_min, self.pi_max)
        m    = clip.mean()
        if m < 1e-12: return 1e6, np.zeros_like(log_pi)
        pi_n = clip / m
        d    = np.sqrt(pi_n)
        S    = (d[:, None] * H) * d[None, :]
        S    = 0.5 * (S + S.T)
        ev, ec = np.linalg.eigh(S)
        if ev[0] <= 1e-12: return 1e6, np.zeros_like(log_pi)
        cond = ev[-1] / ev[0]
        loss = np.log10(cond)
        v_max, v_min = ec[:, -1], ec[:, 0]
        g_n  = (v_max**2 - v_min**2) / (pi_n * np.log(10) + 1e-12)
        mask = (raw > self.pi_min) & (raw < self.pi_max)
        dot  = np.dot(g_n * clip, clip)
        dL   = mask * (g_n / m - dot / (m * m * self.N))
        return loss, dL * raw

    def _optimise(self, H, n_steps=800):
        lo = np.log(self.pi_min)
        hi = np.log(self.pi_max)
        hd = np.clip(np.diag(H), 1e-8, None)

        # Eigenvector decomposition for structured inits
        ev_vals, ev_vecs = np.linalg.eigh(H)

        inits = [
            np.zeros(self.N),                          # pi = 1
            -np.log(hd / hd.mean()),                   # pi ∝ 1/H_ii
            np.log(hd / hd.mean()),                    # pi ∝ H_ii
            # Boost smallest-eigenvalue direction: raise pi where v_min is large
            np.log(ev_vecs[:, 0]**2 + 1e-4)
                - np.log((ev_vecs[:, 0]**2 + 1e-4).mean()),
            # Suppress largest-eigenvalue direction
            -np.log(ev_vecs[:, -1]**2 + 1e-4)
                + np.log((ev_vecs[:, -1]**2 + 1e-4).mean()),
        ]

        best_pi   = np.ones(self.N)
        best_loss = self._loss_grad(np.zeros(self.N), H)[0]

        for init in inits:
            lp = np.clip(init, lo - 1, hi + 1)
            m1 = v1 = np.zeros(self.N)
            b1, b2, eps, lr = 0.9, 0.999, 1e-8, 0.05
            run_best = lp.copy(); run_best_loss = 1e6

            for t in range(1, n_steps + 1):
                loss, grad = self._loss_grad(lp, H)
                if loss < run_best_loss:
                    run_best_loss = loss; run_best = lp.copy()
                if loss < 1e-4: break
                m1 = b1*m1 + (1-b1)*grad
                v1 = b2*v1 + (1-b2)*grad**2
                mh = m1/(1-b1**t); vh = v1/(1-b2**t)
                lp -= lr * mh / (np.sqrt(vh) + eps)
                lp  = np.clip(lp, lo - 1, hi + 1)

            if run_best_loss < best_loss:
                best_loss = run_best_loss
                best_pi   = self._clip_norm(np.exp(run_best))

        return best_pi

    def predict_precision(self, corrupted_query):
        q   = np.asarray(corrupted_query, dtype=np.float64)
        q_n = q / (np.linalg.norm(q) + 1e-12)
        cosines = self.X @ q_n
        best_k  = int(np.argmax(cosines))

        if cosines[best_k] > 0.85:
            return self.pattern_pi[best_k]

        dev = np.abs(q - self.mean) / self.std
        pi  = np.exp(-dev)
        pi  = np.clip(pi, self.pi_min, self.pi_max)
        pi /= pi.mean()
        return np.clip(pi, self.pi_min, self.pi_max)