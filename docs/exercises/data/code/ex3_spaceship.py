"""Exercício 3 — Preparando dados do mundo real (Spaceship Titanic) para tanh.

Lê data/spaceship-titanic/train.csv (raiz do repositório; baixe do Kaggle:
https://www.kaggle.com/competitions/spaceship-titanic), descreve o dataset,
separa treino/teste ANTES de qualquer estatística, imputa, codifica, cria
TotalSpend, aplica log(1 + x) aos gastos, escala para [-1, 1] e verifica.

Uso:  python ex3_spaceship.py   (de qualquer diretório)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent.parent  # docs/exercises/data/code -> raiz do repo
FIGURES = HERE.parent / "figures"
FIGURES.mkdir(exist_ok=True)
DATA = ROOT / "data" / "spaceship-titanic" / "train.csv"

SEED = 42
rng = np.random.default_rng(SEED)

pd.set_option("display.width", 120)

# ---------------------------------------------------------------------------
# A — Conheça os dados
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA)
print("=== Exercício 3A — visão geral ===")
print(f"shape bruto: {df.shape}")
print(df.dtypes)

n_pos = int(df["Transported"].sum())
print(f"\nTransported: {n_pos} True / {len(df) - n_pos} False -> "
      f"proporção positiva = {n_pos / len(df):.4f} ({n_pos / len(df) * 100:.2f}%)")

SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
NUMERIC = ["Age"] + SPEND
CATEGORICAL = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
DROP = ["PassengerId", "Cabin", "Name"]
TARGET = "Transported"
print(f"\nnuméricas   : {NUMERIC}")
print(f"categóricas : {CATEGORICAL}")
print(f"descartadas : {DROP}  (identificadores / texto livre)")

missing = pd.DataFrame({
    "faltantes": df.isna().sum(),
    "percentual": (df.isna().mean() * 100).round(2),
})
print("\n=== Exercício 3A — valores faltantes por coluna ===")
print(missing.to_markdown())
print(f"total de células faltantes: {int(df.isna().sum().sum())}; "
      f"linhas com pelo menos um NaN: {int(df.isna().any(axis=1).sum())}")

stats = df[SPEND].agg(["mean", "median", "max"]).T.round(2)
stats["fração_zero"] = (df[SPEND] == 0).mean().round(3)
print("\n=== Exercício 3A — gastos: média, mediana, máximo (dataset completo) ===")
print(stats.to_markdown())

# ---------------------------------------------------------------------------
# B — Separe antes de transformar
# ---------------------------------------------------------------------------
X = df.drop(columns=DROP + [TARGET])
y = df[TARGET].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)
print("\n=== Exercício 3B — split estratificado 80/20 ===")
print(f"treino: {X_train.shape}, positivos = {y_train.mean():.4f}")
print(f"teste : {X_test.shape}, positivos = {y_test.mean():.4f}")

fc_train = X_train["FoodCourt"]
print(f"FoodCourt no TREINO (antes de transformar): média = {fc_train.mean():.2f}, "
      f"mediana = {fc_train.median():.2f}, máximo = {fc_train.max():.0f}")

# ---------------------------------------------------------------------------
# C — Pré-processe (tudo ajustado no treino, aplicado no teste)
# ---------------------------------------------------------------------------
# 1. Dados faltantes.
#    numéricas   -> mediana (robusta à cauda pesada dos gastos; para os gastos a
#                   mediana é 0, o valor mais plausível para quem não consumiu);
#    categóricas -> categoria mais frequente (moda), ajustada no treino.
num_imputer = SimpleImputer(strategy="median").fit(X_train[NUMERIC])
cat_imputer = SimpleImputer(strategy="most_frequent").fit(X_train[CATEGORICAL].astype(object))
print("\n=== Exercício 3C.1 — valores de imputação (aprendidos no treino) ===")
print(dict(zip(NUMERIC, num_imputer.statistics_)))
print(dict(zip(CATEGORICAL, cat_imputer.statistics_)))


def impute(Xp):
    out = Xp.copy()
    out[NUMERIC] = num_imputer.transform(Xp[NUMERIC])
    out[CATEGORICAL] = cat_imputer.transform(Xp[CATEGORICAL].astype(object))
    return out


Xtr = impute(X_train)
Xte = impute(X_test)

# 2. Features categóricas -> one-hot. handle_unknown="ignore": uma categoria
#    que só aparece no teste vira um vetor todo-zero (nenhuma coluna nova é
#    criada; as colunas são fixadas pelo treino). drop="if_binary" evita a
#    coluna redundante em CryoSleep/VIP (True/False).
encoder = OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)
encoder.fit(Xtr[CATEGORICAL].astype(str))
onehot_cols = list(encoder.get_feature_names_out(CATEGORICAL))
print("\n=== Exercício 3C.2 — colunas one-hot ===")
print(onehot_cols)
# demonstração do tratamento de categoria não vista
demo = pd.DataFrame([["Pluto", "False", "TRAPPIST-1e", "False"]], columns=CATEGORICAL)
print("categoria não vista ('Pluto' em HomePlanet) ->", encoder.transform(demo)[0][:3], "(tudo zero)")

# 3. Engenharia de features: TotalSpend = soma dos cinco gastos (já imputados).
for d in (Xtr, Xte):
    d["TotalSpend"] = d[SPEND].sum(axis=1)
NUM_FINAL = NUMERIC + ["TotalSpend"]

# 4. Cauda pesada: log(1 + x) nos gastos (e no TotalSpend, que é a soma deles).
LOG_COLS = SPEND + ["TotalSpend"]
fc_raw_train = Xtr["FoodCourt"].to_numpy().copy()  # guardado para a Figura 6
for d in (Xtr, Xte):
    d[LOG_COLS] = np.log1p(d[LOG_COLS])
fc_log_train = Xtr["FoodCourt"].to_numpy().copy()

# 5. Escalonamento: normalização Min-Max para [-1, 1], ajustada no treino.
scaler = MinMaxScaler(feature_range=(-1, 1)).fit(Xtr[NUM_FINAL])


def assemble(d):
    num = scaler.transform(d[NUM_FINAL])
    cat = encoder.transform(d[CATEGORICAL].astype(str))
    return pd.DataFrame(np.hstack([num, cat]), columns=NUM_FINAL + onehot_cols, index=d.index)


F_train = assemble(Xtr)
F_test = assemble(Xte)

# ---------------------------------------------------------------------------
# D — Verifique e visualize
# ---------------------------------------------------------------------------
fc_final_train = F_train["FoodCourt"].to_numpy()
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].hist(fc_raw_train, bins=50, color="tab:red", alpha=0.8)
axes[0].set_title("Antes: FoodCourt bruto (treino)")
axes[0].set_xlabel("gasto em FoodCourt")
axes[1].hist(fc_log_train, bins=50, color="tab:orange", alpha=0.8)
axes[1].set_title("Após log(1 + x)")
axes[1].set_xlabel("log(1 + FoodCourt)")
axes[2].hist(fc_final_train, bins=50, color="tab:green", alpha=0.8)
axes[2].set_title("Depois: log(1 + x) + Min-Max para [-1, 1]")
axes[2].set_xlabel("FoodCourt pré-processado")
for ax in axes:
    ax.set_ylabel("contagem de passageiros (treino)")
    ax.grid(alpha=0.3)
fig.suptitle("Figura 6 — FoodCourt antes e depois do pré-processamento (conjunto de treino)")
fig.tight_layout()
fig.savefig(FIGURES / "fig6.png", dpi=150)
plt.close(fig)

print("\n=== Exercício 3D — checagens finais ===")
print(f"NaN remanescentes: treino = {int(F_train.isna().sum().sum())}, teste = {int(F_test.isna().sum().sum())}")
print(f"shape final: treino = {F_train.shape}, teste = {F_test.shape}")
print(f"treino: min = {F_train.to_numpy().min():.4f}, max = {F_train.to_numpy().max():.4f}")
print(f"teste : min = {F_test.to_numpy().min():.4f}, max = {F_test.to_numpy().max():.4f}")
print("colunas numéricas do teste fora de [-1, 1]:",
      {c: (round(F_test[c].min(), 3), round(F_test[c].max(), 3))
       for c in NUM_FINAL if F_test[c].min() < -1 or F_test[c].max() > 1})
# O Min-Max só garante [-1, 1] no treino; um valor de teste maior que o máximo
# do treino ultrapassa 1. Para respeitar a faixa da tanh, recortamos o teste.
n_clipped = int(((F_test[NUM_FINAL] < -1) | (F_test[NUM_FINAL] > 1)).sum().sum())
F_test[NUM_FINAL] = F_test[NUM_FINAL].clip(-1, 1)
print(f"após clip do teste em [-1, 1] ({n_clipped} valores recortados): "
      f"min = {F_test.to_numpy().min():.4f}, max = {F_test.to_numpy().max():.4f}")
print("\nprimeiras linhas da matriz final de treino:")
print(F_train.head(3).round(3).to_string())
print("\nFiguras salvas em", FIGURES)
