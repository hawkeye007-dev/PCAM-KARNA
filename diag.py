# diag.py – run in bench-p04-pcam root
import numpy as np
from data import make_patterns
from pcam_model import PCAMModel, build_default_R

seed = 42
X = make_patterns(K=16, N=64, seed=seed)
R = build_default_R(N=64, seed=seed)
model = PCAMModel(X, R)

print("=== PATTERN STRUCTURE ===")
for k in range(4):
    for j in range(k+1, min(k+4, 16)):
        cos = np.dot(X[k], X[j])
        print(f"  cos({k},{j}) = {cos:.6f}")

print("\n=== EQUILIBRIUM vs STORED PATTERN ===")
for k in range(3):
    a_star = model.find_equilibrium(X[k])
    cos_to_x = np.dot(a_star, X[k]) / (np.linalg.norm(a_star) * np.linalg.norm(X[k]))
    print(f"  Pattern {k}: |a*|={np.linalg.norm(a_star):.4f}, cos(a*, x_k)={cos_to_x:.6f}")
    # Hessian at a*
    H = model.hessian(a_star)
    H = 0.5 * (H + H.T)
    diag = np.diag(H)
    print(f"    H diag: min={diag.min():.6f}, max={diag.max():.6f}, mean={diag.mean():.6f}")
    eigvals, eigvecs = np.linalg.eigh(H)
    print(f"    H eigvals: min={eigvals[0]:.4f}, max={eigvals[-1]:.4f}, cond={eigvals[-1]/eigvals[0]:.4f}")
    v_max = eigvecs[:, -1]
    uniformity = np.dot(np.abs(v_max), np.ones(64)/8)  # should be 1 if uniform
    print(f"    Dominant eigvec uniformity: {uniformity:.6f} (1.0 = perfectly uniform)")
    
    # Test our eigenvector-based approach
    pi_raw = 1.0 / (np.abs(v_max)**2 + 1e-8)
    pi = np.clip(pi_raw, 0.1, 10.0)
    pi /= pi.mean()
    pi = np.clip(pi, 0.1, 10.0)
    d = np.sqrt(pi)
    S = d[:, None] * H * d[None, :]
    S = 0.5 * (S + S.T)
    eigs_S = np.linalg.eigvalsh(S)
    print(f"    After eigenvector pi: cond={eigs_S.max()/eigs_S.min():.4f}")
    
    # What does the harness actually do?
    pi_I = np.ones(64)
    dI = np.sqrt(pi_I)
    SI = dI[:, None] * H * dI[None, :]
    eigs_SI = np.linalg.eigvalsh(SI)
    print(f"    Baseline (pi=I): cond={eigs_SI.max()/eigs_SI.min():.4f}")
    
print("\n=== DIRECT SPREAD COMPARISON ===")
# Exactly replicate what metrics.py does
for k in [0, 1]:
    pattern = X[k]
    a_star = model.find_equilibrium(pattern)
    H = model.hessian(a_star)
    H = 0.5 * (H + H.T)
    
    # Our pi
    _, eigvecs = np.linalg.eigh(H)
    v_max = np.abs(eigvecs[:, -1])
    pi_raw = 1.0 / (v_max**2 + 1e-8)
    pi_agent = model.clip_and_normalise(pi_raw)
    pi_I = np.ones(64)
    
    def spread(pi, H):
        d = np.sqrt(np.clip(pi, 1e-12, None))
        S = d[:, None] * H * d[None, :]
        S = 0.5 * (S + S.T)
        eigs = np.linalg.eigvalsh(S)
        eigs = eigs[eigs > 1e-9]
        return eigs.max() / eigs.min()
    
    s_base = spread(pi_I, H)
    s_agent = spread(pi_agent, H)
    print(f"  Pattern {k}: baseline={s_base:.4f}, agent={s_agent:.4f}, reduction={s_base/s_agent:.2f}x")