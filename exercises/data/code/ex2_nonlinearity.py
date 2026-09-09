"""Exercício 2 — Não-linearidade em 5D.

Dataset I: duas gaussianas multivariadas em R^5 com covariâncias diferentes.
Dataset II: duas cascas concêntricas em R^5 (direções uniformes na esfera,
raio gaussiano). Projeta com PCA (Figura 4), mede distância entre centros e
histogramas de raio em 5D (Figura 5), e verifica uma função não linear que
separa o Dataset II.

Uso:  python ex2_nonlinearity.py   (de qualquer diretório)
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  # scikit-learn só para PCA

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

rng = np.random.default_rng(42)
N = 500  # amostras por classe

# ---------------------------------------------------------------------------
# A — Dataset I: gaussianas multivariadas
# ---------------------------------------------------------------------------
mu_A = np.zeros(5)
Sigma_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])
mu_B = np.full(5, 1.5)
Sigma_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5],
])

XA = rng.multivariate_normal(mu_A, Sigma_A, size=N)
XB = rng.multivariate_normal(mu_B, Sigma_B, size=N)
X_I = np.vstack([XA, XB])
y_I = np.array([0] * N + [1] * N)  # 0 = Classe A, 1 = Classe B

# ---------------------------------------------------------------------------
# B — Dataset II: cascas concêntricas
# ---------------------------------------------------------------------------
def shell(n, radius_mean, radius_std):
    v = rng.standard_normal(size=(n, 5))          # v ~ N(0, I_5)
    u = v / np.linalg.norm(v, axis=1, keepdims=True)  # direção uniforme na esfera
    rho = rng.normal(radius_mean, radius_std, size=n)  # raio ~ N(media, desvio)
    return rho[:, None] * u                         # x = rho * u


XC = shell(N, 2.0, 0.4)  # Classe C (núcleo)
XD = shell(N, 5.0, 0.4)  # Classe D (casca)
X_II = np.vstack([XC, XD])
y_II = np.array([0] * N + [1] * N)  # 0 = Classe C, 1 = Classe D

# ---------------------------------------------------------------------------
# C — Visualize e compare
# ---------------------------------------------------------------------------
pca_I = PCA(n_components=2).fit(X_I)
pca_II = PCA(n_components=2).fit(X_II)
P_I = pca_I.transform(X_I)
P_II = pca_II.transform(X_II)
ev_I = pca_I.explained_variance_ratio_
ev_II = pca_II.explained_variance_ratio_

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, P, yy, names, title, ev in [
    (axes[0], P_I, y_I, ["Classe A", "Classe B"], "Dataset I — gaussianas", ev_I),
    (axes[1], P_II, y_II, ["Classe C (núcleo)", "Classe D (casca)"], "Dataset II — cascas", ev_II),
]:
    for k, (name, col) in enumerate(zip(names, ["tab:blue", "tab:orange"])):
        ax.scatter(P[yy == k, 0], P[yy == k, 1], s=12, alpha=0.6, color=col, label=name)
    ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}% da variância)")
    ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f}% da variância)")
    ax.set_title(f"{title} — PCA 2D (PC1+PC2 = {ev.sum() * 100:.1f}%)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
fig.suptitle("Figura 4 — Projeção PCA em 2 dimensões dos dois datasets 5D")
fig.tight_layout()
fig.savefig(FIGURES / "fig4.png", dpi=150)
plt.close(fig)

print("=== Exercício 2C — variância explicada (PC1 + PC2) ===")
print(f"Dataset I : PC1 = {ev_I[0]:.4f}, PC2 = {ev_I[1]:.4f}, soma = {ev_I.sum():.4f}")
print(f"Dataset II: PC1 = {ev_II[0]:.4f}, PC2 = {ev_II[1]:.4f}, soma = {ev_II.sum():.4f}")

# Medidas geométricas em 5D (sem redução)
def center_distance(X, yy):
    return float(np.linalg.norm(X[yy == 0].mean(axis=0) - X[yy == 1].mean(axis=0)))


d_I = center_distance(X_I, y_I)
d_II = center_distance(X_II, y_II)
r_I = np.linalg.norm(X_I, axis=1)
r_II = np.linalg.norm(X_II, axis=1)

print("\n=== Exercício 2C — distância entre os centros (5D) ===")
print(f"Dataset I : ||mu_A - mu_B|| = {d_I:.4f}  (nominal: {np.linalg.norm(mu_A - mu_B):.4f})")
print(f"Dataset II: ||mu_C - mu_D|| = {d_II:.4f}")
for name, rr, yy in [("I", r_I, y_I), ("II", r_II, y_II)]:
    print(f"Dataset {name}: raio médio classe 0 = {rr[yy == 0].mean():.3f} "
          f"(min {rr[yy == 0].min():.3f}, max {rr[yy == 0].max():.3f}); "
          f"classe 1 = {rr[yy == 1].mean():.3f} "
          f"(min {rr[yy == 1].min():.3f}, max {rr[yy == 1].max():.3f})")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
bins_I = np.linspace(0, max(r_I.max(), 1), 40)
axes[0].hist(r_I[y_I == 0], bins=bins_I, alpha=0.6, color="tab:blue", label="Classe A")
axes[0].hist(r_I[y_I == 1], bins=bins_I, alpha=0.6, color="tab:orange", label="Classe B")
axes[0].set_title(f"Dataset I — raio ‖x‖ em 5D (distância entre centros = {d_I:.2f})")
bins_II = np.linspace(0, r_II.max() + 0.2, 40)
axes[1].hist(r_II[y_II == 0], bins=bins_II, alpha=0.6, color="tab:blue", label="Classe C (núcleo)")
axes[1].hist(r_II[y_II == 1], bins=bins_II, alpha=0.6, color="tab:orange", label="Classe D (casca)")
axes[1].set_title(f"Dataset II — raio ‖x‖ em 5D (distância entre centros = {d_II:.2f})")
for ax in axes:
    ax.set_xlabel("raio ‖x‖")
    ax.set_ylabel("contagem de pontos")
    ax.legend()
    ax.grid(alpha=0.3)
fig.suptitle("Figura 5 — Histograma do raio de cada ponto, classes sobrepostas")
fig.tight_layout()
fig.savefig(FIGURES / "fig5.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# D — Análise: uma função simples das entradas que separa o Dataset II
# ---------------------------------------------------------------------------
# f(x) = sum_i x_i^2 = ||x||^2.  Limiar no ponto médio entre os raios (2 e 5):
# 3.5^2 = 12.25.  Classe D  <=>  f(x) > 12.25.
threshold = 3.5 ** 2
f = (X_II ** 2).sum(axis=1)
pred = (f > threshold).astype(int)
acc = float((pred == y_II).mean())
print("\n=== Exercício 2D — separador não linear f(x) = ||x||^2 ===")
print(f"f(x) > {threshold} classifica corretamente {acc * 100:.2f}% do Dataset II")
print(f"f mínimo classe D = {f[y_II == 1].min():.3f}; f máximo classe C = {f[y_II == 0].max():.3f}")

# E a melhor reta possível no plano PCA do Dataset II? Um hiperplano só pode
# usar uma combinação linear w.x + b; como as duas classes têm o MESMO centro
# (~0) e são simétricas, qualquer hiperplano pelo centro deixa metade de cada
# classe de cada lado. Verificamos isso numericamente com o eixo PC1.
for name, P, yy in [("I", P_I, y_I), ("II", P_II, y_II)]:
    best = 0.0
    for t in np.linspace(P[:, 0].min(), P[:, 0].max(), 400):
        acc_t = max(((P[:, 0] > t) == yy).mean(), ((P[:, 0] <= t) == yy).mean())
        best = max(best, acc_t)
    print(f"Dataset {name}: melhor corte linear ao longo de PC1 acerta {best * 100:.2f}%")

print("\nFiguras salvas em", FIGURES)
