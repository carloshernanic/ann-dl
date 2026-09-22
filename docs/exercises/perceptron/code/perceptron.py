"""Perceptron de camada única, escrito do zero (apenas NumPy).

Este módulo é usado, SEM alteração, pelos dois exercícios do relatório
(`run_exercises.py`). Contém:

- a ativação degrau `step`;
- a predição  y_hat = step(w . x + b);
- a regra de atualização dirigida pelo erro, com rótulos {0, 1}:
      w <- w + eta * (y - y_hat) * x,   b <- b + eta * (y - y_hat)
- o laço de treinamento com o critério de parada do enunciado (época sem
  nenhuma atualização, ou 100 épocas), registrando a acurácia por época;
- o "bolso" (pocket algorithm): a única coisa adicionada ao laço é a cópia
  de (w, b) sempre que uma atualização produz a maior acurácia já vista.
"""
import numpy as np


def step(z):
    """Ativação degrau: 1 se z >= 0, senão 0 (funciona para escalar ou vetor)."""
    return (z >= 0).astype(int)


class Perceptron:
    """Perceptron de uma camada com a regra de atualização para rótulos {0, 1}."""

    def __init__(self, rng, eta=0.01, max_epochs=100, w0=None):
        self.rng = rng
        self.eta = eta
        self.max_epochs = max_epochs
        # Inicialização NÃO nula: w ~ N(0, 0.01), b = 0.  Um w0 explícito
        # permite repetir um treino mudando SOMENTE eta (item D.2) ou partir
        # de w = 0 (item D.3).
        self.w = rng.normal(0.0, 0.01, size=2) if w0 is None else np.array(w0, dtype=float)
        self.b = 0.0

    # ------------------------------------------------------------------ predição
    def predict(self, X):
        """y_hat = step(w . x + b) para cada linha de X."""
        return step(X @ self.w + self.b)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))

    # ------------------------------------------------------------------- treino
    def fit(self, X, y, shuffle=False):
        """Treina até uma época sem atualização ou até max_epochs.

        As amostras são visitadas na ordem em que estão em X (o script
        embaralha o dataset uma única vez, logo após gerá-lo, para que todos os
        treinos vejam exatamente a mesma sequência). Com shuffle=True a ordem
        é re-sorteada a cada época com o rng.

        Registra, por época: a acurácia dos pesos correntes, a melhor acurácia
        vista até então (bolso) e o número de atualizações da época.
        Retorna um dicionário com o histórico e os pesos do bolso.
        """
        n = len(X)
        self.history = {"acc": [], "pocket_acc": [], "updates": []}

        # Bolso: melhor (w, b) já visto, avaliado no dataset completo.
        self.pocket_w = self.w.copy()
        self.pocket_b = self.b
        self.pocket_acc = self.accuracy(X, y)
        self.pocket_epoch = 0          # época (1-based) em que o melhor ocorreu
        self.pocket_update = 0         # índice global da atualização que o produziu
        n_updates_total = 0

        self.epochs_run = 0
        for epoch in range(1, self.max_epochs + 1):
            order = self.rng.permutation(n) if shuffle else np.arange(n)
            n_updates = 0
            for i in order:
                x_i, y_i = X[i], y[i]
                y_hat = step(x_i @ self.w + self.b)      # predição para UMA amostra
                error = y_i - y_hat                       # 0, +1 ou -1
                if error != 0:                            # só erros atualizam
                    self.w = self.w + self.eta * error * x_i
                    self.b = self.b + self.eta * error
                    n_updates += 1
                    n_updates_total += 1
                    # --- bolso: a única adição ao laço ---------------------------
                    acc_now = self.accuracy(X, y)
                    if acc_now > self.pocket_acc:
                        self.pocket_acc = acc_now
                        self.pocket_w = self.w.copy()
                        self.pocket_b = self.b
                        self.pocket_epoch = epoch
                        self.pocket_update = n_updates_total
            self.epochs_run = epoch
            self.history["acc"].append(self.accuracy(X, y))
            self.history["pocket_acc"].append(self.pocket_acc)
            self.history["updates"].append(n_updates)
            if n_updates == 0:          # época inteira sem erro: convergiu
                self.converged = True
                break
        else:
            self.converged = False
        return self.history
