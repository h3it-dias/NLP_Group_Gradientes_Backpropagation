"""Demonstrações práticas do motor de autograd escalar: um neurônio manual,
o mesmo neurônio com tanh 'expandido' via exp, checagem cruzada com o
autograd real do PyTorch, e um MLP treinado por gradiente descendente manual."""
from __future__ import annotations

from pathlib import Path

from .engine import Value
from .nn import MLP
from .visualize import draw_dot

OUTPUT_DIR = Path(__file__).parent / "diagrams"


def single_neuron_demo() -> Value:
    """Um neurônio: x1,x2 -> soma ponderada com w1,w2,b -> tanh."""
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")
    b = Value(6.8813735870195432, label="b")

    x1w1 = x1 * w1
    x1w1.label = "x1*w1"
    x2w2 = x2 * w2
    x2w2.label = "x2*w2"
    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = "x1*w2 + x2*w2"
    n = x1w1x2w2 + b
    n.label = "n"
    o = n.tanh()
    o.label = "o"
    o.backward()
    return o


def expanded_tanh_neuron_demo() -> Value:
    """Mesmo neurônio, mas com tanh decomposto em exp() na mão — confirma
    que o resultado (e o gradiente) não muda ao expandir a operação."""
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")
    b = Value(6.8813735870195432, label="b")

    x1w1 = x1 * w1
    x1w1.label = "x1*w1"
    x2w2 = x2 * w2
    x2w2.label = "x2*w2"
    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = "x1*w2 + x2*w2"
    n = x1w1x2w2 + b
    n.label = "n"

    e = (2 * n).exp()
    o = (e - 1) / (e + 1)
    o.label = "o"
    o.backward()
    return o


def pytorch_cross_check_demo() -> None:
    """O mesmo neurônio, agora com torch.Tensor(requires_grad=True), pra
    comparar os gradientes calculados à mão com os do autograd de verdade."""
    import torch

    x1 = torch.tensor([2.0], requires_grad=True, dtype=torch.double)
    x2 = torch.tensor([0.0], requires_grad=True, dtype=torch.double)
    w1 = torch.tensor([-3.0], requires_grad=True, dtype=torch.double)
    w2 = torch.tensor([1.0], requires_grad=True, dtype=torch.double)
    b = torch.tensor([6.8813735870195432], requires_grad=True, dtype=torch.double)

    n = x1 * w1 + x2 * w2 + b
    o = torch.tanh(n)

    print(f"o = {o.data.item()}")
    o.backward()

    print(f"x2.grad = {x2.grad.item()}")
    print(f"w2.grad = {w2.grad.item()}")
    print(f"x1.grad = {x1.grad.item()}")
    print(f"w1.grad = {w1.grad.item()}")


def mlp_training_demo(epochs: int = 40, lr: float = 0.05) -> list[Value]:
    """Treina um MLP(3, [4,4,1]) num dataset de 4 exemplos via MSE +
    gradiente descendente manual — sem otimizador do PyTorch, o passo de
    update é literalmente p.data += -lr * p.grad."""
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    n = MLP(3, [4, 4, 1])

    for k in range(epochs):
        ypred = [n(x) for x in xs]
        loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

        for p in n.parameters():
            p.grad = 0.0
        loss.backward()

        for p in n.parameters():
            p.data += -lr * p.grad

        print(f"epoch {k:>2d} | loss {loss.data:.6f}")

    return [n(x) for x in xs]


def _print_section(title: str) -> None:
    separator = "-" * 60
    print(f"\n{separator}\n{title}\n{separator}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    _print_section("Neurônio manual (tanh)")
    o = single_neuron_demo()
    print(f"o = {o.data:.4f}")
    path = draw_dot(o).render(directory=OUTPUT_DIR, filename="single_neuron", format="png", cleanup=True)
    print(f"Grafo salvo em {path}")

    _print_section("Neurônio com tanh expandido via exp")
    o = expanded_tanh_neuron_demo()
    print(f"o = {o.data:.4f}")
    path = draw_dot(o).render(directory=OUTPUT_DIR, filename="expanded_tanh_neuron", format="png", cleanup=True)
    print(f"Grafo salvo em {path}")

    _print_section("Checagem cruzada com PyTorch")
    pytorch_cross_check_demo()

    _print_section("Treinando um MLP com gradiente descendente manual")
    mlp_training_demo()
