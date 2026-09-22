"""Exercícios 1 e 2 — Perceptron em dados separáveis e em dados sobrepostos.

Gera os dois datasets, treina o perceptron de `perceptron.py` (sem alteração
entre os exercícios), produz as Figuras 1 a 6 em ../figures e imprime todos os
números citados no relatório.

Uso:  python run_exercises.py   (de qualquer diretório)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from perceptron import Perceptron  # noqa: E402  (implementação própria)

FIGURES = HERE.parent / "figures"
FIGURES.mkdir(exist_ok=True)

rng = np.random.default_rng(42)  # o MESMO rng em todo o relatório
N = 1000                          # amostras por classe
COLORS = {0: "tab:blue", 1: "tab:orange"}


# ---------------------------------------------------------------------------
# utilidades comuns
# ---------------------------------------------------------------------------
def make_data(mean0, mean1, cov):
    """Duas classes gaussianas 2D (1000 cada), embaralhadas UMA vez."""
    X0 = rng.multivariate_normal(mean0, cov, size=N)
    X1 = rng.multivariate_normal(mean1, cov, size=N)
    X = np.vstack([X0, X1])
    y = np.array([0] * N + [1] * N)
    perm = rng.permutation(len(X))   # ordem fixa de visita para todos os treinos
    return X[perm], y[perm]


def scatter(ax, X, y, alpha=0.5, size=10):
    for k in (0, 1):
        ax.scatter(X[y == k, 0], X[y == k, 1], s=size, alpha=alpha,
                   color=COLORS[k], label=f"Classe {k}")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.grid(alpha=0.3)


def boundary_xy(w, b, xlim):
    """Pontos da reta w.x + b = 0 dentro de xlim (resolvendo para x2)."""
    xs = np.array(xlim)
    if abs(w[1]) > 1e-12:
        return xs, -(w[0] * xs + b) / w[1]
    x_vert = -b / w[0]
    return np.array([x_vert, x_vert]), np.array([-1e3, 1e3])


def mark_wrong(ax, X, y, w, b, label, marker="x", color="black"):
    wrong = ((X @ w + b >= 0).astype(int)) != y
    ax.scatter(X[wrong, 0], X[wrong, 1], marker=marker, s=40, color=color,
               linewidths=1.0, label=f"{label} ({wrong.sum()} erros)")
    return int(wrong.sum())


def fmt(v):
    return np.array2string(np.asarray(v), precision=4, separator=", ")


# ===========================================================================
# Exercício 1 — dados separáveis
# ===========================================================================
print("=" * 70)
print("EXERCÍCIO 1 — dados separáveis")
print("=" * 70)

cov1 = [[0.5, 0.0], [0.0, 0.5]]
X1, y1 = make_data([1.5, 1.5], [5.0, 5.0], cov1)

# Figura 1 -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
scatter(ax, X1, y1)
ax.set_title("Figura 1 — Exercício 1: duas classes separáveis (1000 pontos cada)")
ax.legend()
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(FIGURES / "fig1.png", dpi=150)
plt.close(fig)

# Treino com eta = 0.01 ------------------------------------------------------
w0 = rng.normal(0.0, 0.01, size=2)   # inicialização sorteada UMA vez (item B)
print(f"w inicial (N(0, 0.01)): {fmt(w0)}, b inicial = 0")

p1 = Perceptron(rng, eta=0.01, max_epochs=100, w0=w0)
p1.fit(X1, y1)
acc1 = p1.accuracy(X1, y1)
print(f"\n[eta = 0.01] w final = {fmt(p1.w)}, b final = {p1.b:.4f}")
print(f"[eta = 0.01] épocas = {p1.epochs_run} (convergiu: {p1.converged}), "
      f"acurácia final = {acc1:.4f} ({acc1 * 100:.2f}%)")
print(f"[eta = 0.01] atualizações por época: {p1.history['updates']}")
print(f"[eta = 0.01] acurácia por época: {[round(a, 4) for a in p1.history['acc']]}")
print(f"[eta = 0.01] direção w/||w|| = {fmt(p1.w / np.linalg.norm(p1.w))}, ||w|| = {np.linalg.norm(p1.w):.4f}")

# Figura 2 -------------------------------------------------------------------
xlim = (X1[:, 0].min() - 0.5, X1[:, 0].max() + 0.5)
ylim = (X1[:, 1].min() - 0.5, X1[:, 1].max() + 0.5)
fig, ax = plt.subplots(figsize=(7, 6))
scatter(ax, X1, y1)
bx, by = boundary_xy(p1.w, p1.b, xlim)
ax.plot(bx, by, "k-", linewidth=2, label="fronteira $w \\cdot x + b = 0$")
n_wrong1 = mark_wrong(ax, X1, y1, p1.w, p1.b, "mal classificados")
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_title(f"Figura 2 — Fronteira de decisão aprendida (η = 0.01, {p1.epochs_run} épocas)")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(FIGURES / "fig2.png", dpi=150)
plt.close(fig)

# Figura 3 -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
epochs = np.arange(1, p1.epochs_run + 1)
ax.plot(epochs, np.array(p1.history["acc"]) * 100, "o-", color="tab:green",
        label="acurácia no dataset completo (fim da época)")
ax.set_xlabel("época")
ax.set_ylabel("acurácia (%)")
ax.set_ylim(0, 105)
ax.set_title("Figura 3 — Exercício 1: acurácia × época (η = 0.01)")
ax.set_xticks(epochs)
ax.grid(alpha=0.3)
# eixo secundário: quantas atualizações (erros) cada época produziu
ax2 = ax.twinx()
ax2.bar(epochs, p1.history["updates"], width=0.25, alpha=0.3, color="tab:red",
        label="atualizações na época")
ax2.set_ylabel("atualizações (erros) na época")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="center right")
fig.tight_layout()
fig.savefig(FIGURES / "fig3.png", dpi=150)
plt.close(fig)

# D.2 — mesma inicialização, mesma ordem, só eta muda ------------------------
p1b = Perceptron(rng, eta=1.0, max_epochs=100, w0=w0)
p1b.fit(X1, y1)
acc1b = p1b.accuracy(X1, y1)
d_small = p1.w / np.linalg.norm(p1.w)
d_big = p1b.w / np.linalg.norm(p1b.w)
angle = np.degrees(np.arccos(np.clip(d_small @ d_big, -1, 1)))
print(f"\n[eta = 1.0 ] w final = {fmt(p1b.w)}, b final = {p1b.b:.4f}")
print(f"[eta = 1.0 ] épocas = {p1b.epochs_run}, acurácia final = {acc1b:.4f} ({acc1b * 100:.2f}%)")
print(f"[eta = 1.0 ] atualizações por época: {p1b.history['updates']}")
print(f"[eta = 1.0 ] direção w/||w|| = {fmt(d_big)}, ||w|| = {np.linalg.norm(p1b.w):.4f}")
print(f"ângulo entre as direções (0.01 vs 1.0): {angle:.2f} graus")
for name, p in [("eta=0.01", p1), ("eta=1.0", p1b)]:
    c = np.array([3.25, 3.25])  # ponto médio entre as médias
    print(f"  {name}: interceptos: x2 = {-p.b / p.w[1]:.3f} em x1 = 0; "
          f"inclinação = {-p.w[0] / p.w[1]:.3f}; distância do ponto médio (3.25, 3.25) "
          f"à fronteira = {abs(p.w @ c + p.b) / np.linalg.norm(p.w):.3f}")

# D.3 — a partir de w = 0: eta só reescala ------------------------------------
pz1 = Perceptron(rng, eta=0.01, max_epochs=100, w0=[0.0, 0.0])
pz1.fit(X1, y1)
pz2 = Perceptron(rng, eta=1.0, max_epochs=100, w0=[0.0, 0.0])
pz2.fit(X1, y1)
print(f"\n[w0 = 0, eta = 0.01] w = {fmt(pz1.w)}, b = {pz1.b:.4f}, épocas = {pz1.epochs_run}, "
      f"acurácia = {pz1.accuracy(X1, y1):.4f}")
print(f"[w0 = 0, eta = 1.0 ] w = {fmt(pz2.w)}, b = {pz2.b:.4f}, épocas = {pz2.epochs_run}, "
      f"acurácia = {pz2.accuracy(X1, y1):.4f}")
print(f"razão w(eta=1)/w(eta=0.01) = {fmt(pz2.w / pz1.w)}, b: {pz2.b / pz1.b:.1f}  "
      f"(esperado: 100 em todos)  | atualizações por época iguais: "
      f"{pz1.history['updates'] == pz2.history['updates']}")

# ===========================================================================
# Exercício 2 — dados sobrepostos
# ===========================================================================
print("\n" + "=" * 70)
print("EXERCÍCIO 2 — dados sobrepostos")
print("=" * 70)

cov2 = [[1.5, 0.0], [0.0, 1.5]]
X2, y2 = make_data([3.0, 3.0], [4.0, 4.0], cov2)
print(f"||x|| médio no Exercício 2: {np.linalg.norm(X2, axis=1).mean():.3f}")

# Figura 4 -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
scatter(ax, X2, y2)
ax.set_title("Figura 4 — Exercício 2: duas classes sobrepostas (1000 pontos cada)")
ax.legend()
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(FIGURES / "fig4.png", dpi=150)
plt.close(fig)

# Treino: MESMA implementação, eta = 0.01, 100 épocas, com bolso -------------
w0_2 = rng.normal(0.0, 0.01, size=2)
p2 = Perceptron(rng, eta=0.01, max_epochs=100, w0=w0_2)
p2.fit(X2, y2)
acc_final = p2.accuracy(X2, y2)
print(f"w inicial = {fmt(w0_2)}")
print(f"convergiu: {p2.converged}; épocas rodadas = {p2.epochs_run}; "
      f"atualizações na última época = {p2.history['updates'][-1]}; "
      f"total de atualizações = {sum(p2.history['updates'])}")
print(f"FINAL : w = {fmt(p2.w)}, b = {p2.b:.4f}, acurácia = {acc_final:.4f} ({acc_final * 100:.2f}%)")
print(f"BOLSO : w = {fmt(p2.pocket_w)}, b = {p2.pocket_b:.4f}, acurácia = {p2.pocket_acc:.4f} "
      f"({p2.pocket_acc * 100:.2f}%), obtido na época {p2.pocket_epoch} "
      f"(atualização nº {p2.pocket_update})")
print(f"acurácia por época (primeiras 10): {[round(a, 3) for a in p2.history['acc'][:10]]}")
print(f"acurácia por época: min = {min(p2.history['acc']):.3f}, max = {max(p2.history['acc']):.3f}, "
      f"média = {np.mean(p2.history['acc']):.3f}")

# Onde a fronteira final está em relação à nuvem?
c2 = np.array([3.5, 3.5])
for name, w, b in [("final", p2.w, p2.b), ("bolso", p2.pocket_w, p2.pocket_b)]:
    dist = (w @ c2 + b) / np.linalg.norm(w)
    frac_pos = np.mean((X2 @ w + b) >= 0)
    print(f"  {name}: ||w|| = {np.linalg.norm(w):.4f}, b = {b:.4f}, b/||w|| = {b / np.linalg.norm(w):.3f}; "
          f"distância assinada do centro (3.5, 3.5) à fronteira = {dist:.3f}; "
          f"fração de pontos preditos como classe 1 = {frac_pos:.3f}")

# Referência: melhor reta possível (bissetriz perpendicular entre as médias,
# que é a fronteira ótima para gaussianas de mesma covariância): x1 + x2 = 7.
w_ref, b_ref = np.array([1.0, 1.0]), -7.0
acc_ref = np.mean(((X2 @ w_ref + b_ref) >= 0).astype(int) == y2)
print(f"referência: reta x1 + x2 = 7 acerta {acc_ref:.4f} ({acc_ref * 100:.2f}%)")

# Figura 5 -------------------------------------------------------------------
xlim2 = (X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5)
ylim2 = (X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
for ax, (name, w, b, col) in zip(axes, [
    ("pesos finais", p2.w, p2.b, "tab:red"),
    ("pesos do bolso (pocket)", p2.pocket_w, p2.pocket_b, "tab:green"),
]):
    scatter(ax, X2, y2, alpha=0.35)
    bx, by = boundary_xy(w, b, xlim2)
    ax.plot(bx, by, color=col, linewidth=2.5, label=f"fronteira — {name}")
    n_wrong = mark_wrong(ax, X2, y2, w, b, "mal classificados")
    acc = 1 - n_wrong / len(X2)
    ax.set_title(f"{name}: acurácia = {acc * 100:.2f}%")
    ax.set_xlim(xlim2)
    ax.set_ylim(ylim2)
    ax.legend(loc="upper left", fontsize=8)
fig.suptitle("Figura 5 — Exercício 2: fronteiras final e do bolso sobre os dados, erros marcados com ×")
fig.tight_layout()
fig.savefig(FIGURES / "fig5.png", dpi=150)
plt.close(fig)

# Figura 6 -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
ep = np.arange(1, p2.epochs_run + 1)
ax.plot(ep, np.array(p2.history["acc"]) * 100, "-", color="tab:red", alpha=0.8,
        label="acurácia dos pesos correntes (fim de cada época)")
ax.plot(ep, np.array(p2.history["pocket_acc"]) * 100, "-", color="tab:green", linewidth=2.5,
        label="melhor até então (bolso)")
ax.axhline(acc_ref * 100, color="gray", linestyle="--", label=f"referência $x_1 + x_2 = 7$ ({acc_ref * 100:.1f}%)")
ax.set_xlabel("época")
ax.set_ylabel("acurácia (%)")
ax.set_title("Figura 6 — Exercício 2: acurácia corrente × melhor até então, por época")
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES / "fig6.png", dpi=150)
plt.close(fig)

# D.3 — mais épocas / eta menor não resolvem (confirmação numérica) ----------
p2_long = Perceptron(rng, eta=0.01, max_epochs=500, w0=w0_2)
p2_long.fit(X2, y2)
p2_small = Perceptron(rng, eta=0.001, max_epochs=100, w0=w0_2)
p2_small.fit(X2, y2)
print(f"\n[500 épocas, eta=0.01 ] final = {p2_long.accuracy(X2, y2):.4f}, bolso = {p2_long.pocket_acc:.4f} "
      f"(época {p2_long.pocket_epoch}), atualizações na última época = {p2_long.history['updates'][-1]}")
print(f"[100 épocas, eta=0.001] final = {p2_small.accuracy(X2, y2):.4f}, bolso = {p2_small.pocket_acc:.4f} "
      f"(época {p2_small.pocket_epoch}), w = {fmt(p2_small.w)}, b = {p2_small.b:.4f}")

print("\nFiguras salvas em", FIGURES)
