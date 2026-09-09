"""Exercício 1 — Nuvens de pontos: geometria e espalhamento em 2D.

Gera 4 classes gaussianas em 2D, estuda o efeito do fator de escala s sobre o
espalhamento, calcula a razão de separação r_ij e a taxa de mistura, e produz
as Figuras 1, 2 e 3 (mais o esboço das fronteiras sobre a Figura 1).

Uso:  python ex1_clouds.py   (de qualquer diretório)
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

rng = np.random.default_rng(42)  # a MESMA semente/rng em todo o relatório

N_PER_CLASS = 100
MEANS = np.array([[2.0, 3.0], [5.0, 6.0], [8.0, 1.0], [15.0, 4.0]])
STDS = np.array([[0.8, 2.5], [1.2, 1.9], [0.9, 0.9], [0.5, 2.0]])
SCALES = [0.5, 1.0, 2.0, 4.0]
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

# ---------------------------------------------------------------------------
# A — Gere as nuvens
# ---------------------------------------------------------------------------
# Sorteamos UMA vez o ruído padrão Z ~ N(0, 1) de cada classe e construímos
# cada ponto como  x = mu + (s * sigma) * z.  Assim o dataset de s = 1 é
# exatamente o da Figura 1 e os quatro datasets do item B diferem SOMENTE pelo
# espalhamento, com o mesmo ruído subjacente: a comparação fica honesta.
Z = rng.standard_normal(size=(4, N_PER_CLASS, 2))
y = np.repeat(np.arange(4), N_PER_CLASS)  # rótulo de cada linha


def make_dataset(s: float) -> np.ndarray:
    """Retorna X (400 x 2): as 4 classes com desvios padrão multiplicados por s."""
    X = MEANS[:, None, :] + (s * STDS)[:, None, :] * Z  # (4, 100, 2)
    return X.reshape(-1, 2)


def scatter_classes(ax, X, y, centers=True, alpha=0.7):
    for k in range(4):
        pts = X[y == k]
        ax.scatter(pts[:, 0], pts[:, 1], s=16, alpha=alpha, color=COLORS[k],
                   label=f"Classe {k}")
    if centers:
        ax.scatter(MEANS[:, 0], MEANS[:, 1], marker="X", s=180, c="black",
                   edgecolor="white", linewidth=1.2, zorder=5, label="Centro (média)")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")


X1 = make_dataset(1.0)

fig, ax = plt.subplots(figsize=(8, 6))
scatter_classes(ax, X1, y)
ax.set_title("Figura 1 — Quatro nuvens gaussianas em 2D (s = 1) com os centros marcados")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "fig1.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# B — Mais ou menos espalhado
# ---------------------------------------------------------------------------
datasets = {s: make_dataset(s) for s in SCALES}

# Figura 2: 4 subplots com os MESMOS limites de eixo (sharex/sharey), definidos
# pelo dataset mais espalhado (s = 4) para que nada fique cortado.
all_pts = np.vstack(list(datasets.values()))
pad = 1.0
xlim = (all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
ylim = (all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)

fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
for ax, s in zip(axes.ravel(), SCALES):
    scatter_classes(ax, datasets[s], y, alpha=0.6)
    ax.set_title(f"s = {s}")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(alpha=0.3)
axes[0, 0].legend(loc="upper left", fontsize=8)
fig.suptitle("Figura 2 — As mesmas 4 classes com os desvios padrão × s (eixos compartilhados)")
fig.tight_layout()
fig.savefig(FIGURES / "fig2.png", dpi=150)
plt.close(fig)

# Razão de separação r_ij, apenas para s = 1, usando os parâmetros nominais
# (média e desvio padrão do enunciado):  r_ij = ||mu_i - mu_j|| / (sb_i + sb_j),
# com sb_k = (sigma_kx + sigma_ky) / 2.
sigma_bar = STDS.mean(axis=1)
r = {}
for i, j in combinations(range(4), 2):
    dist = np.linalg.norm(MEANS[i] - MEANS[j])
    r[(i, j)] = dist / (sigma_bar[i] + sigma_bar[j])

pair_min = min(r, key=r.get)
r_min = r[pair_min]

print("\n=== Exercício 1B — razão de separação r_ij (s = 1) ===")
print(f"sigma_bar por classe: {np.round(sigma_bar, 3)}")
print("| par (i, j) | ||mu_i - mu_j|| | sb_i + sb_j | r_ij |")
for (i, j), val in r.items():
    d = np.linalg.norm(MEANS[i] - MEANS[j])
    print(f"| ({i}, {j}) | {d:.3f} | {sigma_bar[i] + sigma_bar[j]:.3f} | {val:.3f} |")
print(f"menor r_ij: par {pair_min} = {r_min:.3f}  ->  em s = 2: {r_min / 2:.3f} (r escala com 1/s)")


# Taxa de mistura: fração de pontos cujo centro (média nominal) mais próximo
# NÃO é o da própria classe. Puramente geométrico: nenhum modelo é treinado.
def mixing_rate(X, y):
    d2 = ((X[:, None, :] - MEANS[None, :, :]) ** 2).sum(axis=2)  # (400, 4)
    nearest = d2.argmin(axis=1)
    return float((nearest != y).mean())


mix = {s: mixing_rate(datasets[s], y) for s in SCALES}
print("\n=== Exercício 1B — taxa de mistura ===")
for s in SCALES:
    print(f"s = {s}: taxa de mistura = {mix[s]:.4f} ({mix[s] * 100:.2f} %)  |  menor r_ij = {r_min / s:.3f}")

# Figura 3: taxa de mistura x s
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(SCALES, [mix[s] * 100 for s in SCALES], "o-", color="tab:purple",
        label="taxa de mistura (centro mais próximo ≠ classe)")
for s in SCALES:
    ax.annotate(f"{mix[s] * 100:.1f}%", (s, mix[s] * 100), textcoords="offset points",
                xytext=(6, 6), fontsize=9)
ax.set_xscale("log", base=2)
ax.set_xticks(SCALES)
ax.set_xticklabels([str(s) for s in SCALES])
ax.set_xlabel("fator de escala s (eixo log₂)")
ax.set_ylabel("taxa de mistura (%)")
ax.set_title("Figura 3 — Taxa de mistura × fator de escala s")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "fig3.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# C — Análise: esboço das fronteiras sobre a Figura 1
# ---------------------------------------------------------------------------
# O esboço é a regra de decisão bayesiana com os parâmetros nominais de cada
# classe (gaussianas com eixos independentes): a fronteira que uma rede bem
# treinada tenderia a aproximar. Nada é treinado — só se avalia a densidade.
def log_density(grid_pts):
    """log p_k(x) de cada classe (k = 0..3) em cada ponto do grid, para s = 1."""
    z = (grid_pts[:, None, :] - MEANS[None, :, :]) / STDS[None, :, :]
    return -0.5 * (z ** 2).sum(axis=2) - np.log(STDS).sum(axis=1)[None, :]


xx, yy = np.meshgrid(np.linspace(-3, 19, 500), np.linspace(-8, 14, 500))
grid = np.c_[xx.ravel(), yy.ravel()]
region1 = log_density(grid).argmax(axis=1).reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
cmap = matplotlib.colors.ListedColormap(COLORS)
for ax, s in zip(axes, [1.0, 4.0]):
    Xs = datasets[s]
    z = (grid[:, None, :] - MEANS[None, :, :]) / (s * STDS)[None, :, :]
    region = (-0.5 * (z ** 2).sum(axis=2)).argmax(axis=1).reshape(xx.shape)
    ax.contourf(xx, yy, region, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], cmap=cmap, alpha=0.18)
    ax.contour(xx, yy, region, levels=[0.5, 1.5, 2.5], colors="black", linewidths=1.2)
    scatter_classes(ax, Xs, y, alpha=0.7)
    # pontos que caem na região de outra classe: onde a rede "necessariamente erra"
    zp = (Xs[:, None, :] - MEANS[None, :, :]) / (s * STDS)[None, :, :]
    pred = (-0.5 * (zp ** 2).sum(axis=2)).argmax(axis=1)
    wrong = pred != y
    ax.scatter(Xs[wrong, 0], Xs[wrong, 1], marker="x", s=60, c="black",
               label=f"fora da própria região ({wrong.mean() * 100:.1f}%)")
    ax.set_title(f"s = {s}: fronteiras esboçadas (regra de Bayes com os parâmetros nominais)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(-3, 19)
    ax.set_ylim(-8, 14)
fig.suptitle("Figura 1 (anotada) — Esboço das fronteiras de decisão sobre as nuvens, em s = 1 e s = 4")
fig.tight_layout()
fig.savefig(FIGURES / "fig1_boundaries.png", dpi=150)
plt.close(fig)

print("\nFiguras salvas em", FIGURES)
