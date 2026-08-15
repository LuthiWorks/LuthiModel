"""Does P = I - G(G^T G + eI)^-1 G^T actually satisfy P.G = 0?

The suggestions doc asserts "Because P . G = 0, the projected gradient
exists entirely inside the null space of the core network. This
mathematically guarantees ... exactly zero degradation."
"""
import numpy as np

rng = np.random.default_rng(0)
P_dim, N = 60, 8           # 60 parameters, 8 reference gradients
A = rng.normal(size=(P_dim, N))     # columns are grad_theta f(x_i)
G = A @ A.T                          # the doc's G = sum of outer products
eps = 1e-6

I = np.eye(P_dim)

# --- the doc's projector ---
P_doc = I - G @ np.linalg.inv(G.T @ G + eps * I) @ G.T

# --- the standard orthogonal projector onto range(A)^perp ---
P_std = I - A @ np.linalg.inv(A.T @ A) @ A.T

print(f"params P={P_dim}, reference grads N={N}, rank(G)={np.linalg.matrix_rank(G)}")
print()
print(f"||P_doc @ G||_F = {np.linalg.norm(P_doc @ G):.6e}")
print(f"||P_std @ G||_F = {np.linalg.norm(P_std @ G):.6e}")
print(f"||G||_F         = {np.linalg.norm(G):.6e}")
print()

# How much of a real gradient survives, and does it disturb the core?
g = rng.normal(size=P_dim)
for name, Pm in (("doc", P_doc), ("standard", P_std)):
    gp = Pm @ g
    leak = np.linalg.norm(A.T @ gp)      # component along protected directions
    print(f"{name:>9}: ||A^T P g|| = {leak:.6e}   (0 = no interference) "
          f"||P g||={np.linalg.norm(gp):.4f}")

print()
print("eps sensitivity of the doc's projector (the guarantee needs eps->0,")
print("but eps>0 is exactly what makes the inverse exist):")
for e in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
    Pd = I - G @ np.linalg.inv(G.T @ G + e * I) @ G.T
    print(f"  eps={e:<8.0e} ||P_doc G||_F = {np.linalg.norm(Pd @ G):.6e}"
          f"   ||P_doc||_F = {np.linalg.norm(Pd):.4f}")

print()
print("cost note: the doc inverts a P x P matrix (G^T G).")
print(f"  here P={P_dim}. In Luthi P is the parameter count, ~1e7-1e8,")
print("  so G^T G would have ~1e14-1e16 entries.")
print("  the standard form inverts A^T A, which is N x N -- here 8x8.")
