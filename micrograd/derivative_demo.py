"""Motivação da derivada, antes de qualquer Value entrar em cena: estima a
derivada numericamente pela definição de limite, pra depois comparar com o
que .backward() calcula de forma exata."""
from __future__ import annotations

from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(__file__).parent / "diagrams"


def f(x):
    return 3 * x**2 - 4 * x + 5


def plot_f(output_path: Path = OUTPUT_DIR / "f_x.png") -> Path:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(exist_ok=True)
    xs = np.arange(-5, 5, 0.25)
    ys = f(xs)
    plt.plot(xs, ys)
    plt.savefig(output_path)
    plt.close()
    return output_path


def numerical_derivative_demo(h: float = 1e-8) -> None:
    x = 2 / 3
    slope = (f(x + h) - f(x)) / h
    print(f"f'(x) em x={x:.4f} pela definição de limite: {slope}")


def numerical_partial_derivative_demo(h: float = 1e-4) -> None:
    """Mesma ideia, mas a derivada parcial em relação a 'a' numa expressão
    com 3 variáveis (a*b + c) — perturba só 'a' e mede o quanto a saída muda."""
    a, b, c = 2.0, -3.0, 10.0

    d1 = a * b + c
    a += h
    d2 = a * b + c

    print(f"d1 = {d1}")
    print(f"d2 = {d2}")
    print(f"slope (∂/∂a) = {(d2 - d1) / h}")


if __name__ == "__main__":
    numerical_derivative_demo()
    numerical_partial_derivative_demo()

    path = plot_f()
    print(f"Gráfico de f(x) salvo em {path}")
