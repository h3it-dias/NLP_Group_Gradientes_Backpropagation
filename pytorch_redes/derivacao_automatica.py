"""Fundamentos de derivação automática (autograd) do PyTorch: como o grafo
computacional é construído, como os gradientes são calculados via regra da
cadeia, e o produto Jacobiano usado no backward de tensores não-escalares."""
from __future__ import annotations

import torch


def build_loss_graph() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Monta um exemplo mínimo de regressão logística (uma camada linear +
    BCE) e retorna os tensores já com o grafo computacional montado."""
    x = torch.ones(5)  # tensor de entrada
    y = torch.zeros(3)  # saída esperada
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    z = torch.matmul(x, w) + b
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
    return x, w, b, z, loss


def print_grad_functions(z: torch.Tensor, loss: torch.Tensor) -> None:
    """Todo tensor calculado a partir de operações com requires_grad=True
    guarda em .grad_fn a função que sabe propagar o gradiente pra trás."""
    print(f"Função de gradiente de z: {z.grad_fn}")
    print(f"Função de gradiente de loss: {loss.grad_fn}")


def compute_gradients(loss: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> None:
    """Chama .backward() na raiz do grafo (loss) e imprime os gradientes
    acumulados em .grad de cada parâmetro folha (w e b)."""
    loss.backward()
    print(f"w.grad:\n{w.grad}")
    print(f"b.grad:\n{b.grad}")


def demo_disable_gradient_tracking(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> None:
    """Por padrão, toda operação sobre tensores com requires_grad=True fica
    rastreada. torch.no_grad() desliga esse rastreamento — útil em inferência,
    quando não precisamos calcular gradiente."""
    z = torch.matmul(x, w) + b
    print(f"requires_grad fora do no_grad: {z.requires_grad}")

    with torch.no_grad():
        z = torch.matmul(x, w) + b
    print(f"requires_grad dentro do no_grad: {z.requires_grad}")


def print_computational_graph_explanation() -> None:
    print(
        "Conceitualmente, o autograd mantém um registro dos dados (tensores) e de\n"
        "todas as operações executadas (junto com os novos tensores resultantes) em\n"
        "um grafo acíclico dirigido (DAG) composto por objetos Function. Nesse DAG,\n"
        "as folhas são os tensores de entrada e as raízes são os tensores de saída.\n"
        "Percorrendo esse grafo das raízes até as folhas, é possível calcular os\n"
        "gradientes automaticamente usando a regra da cadeia.\n"
        "\n"
        "Em um forward pass, o autograd faz duas coisas ao mesmo tempo:\n"
        "  - executa a operação pedida para calcular o tensor resultante;\n"
        "  - mantém a função de gradiente da operação no DAG.\n"
        "\n"
        "O backward pass começa quando .backward() é chamado na raiz do DAG. A\n"
        "partir daí o autograd:\n"
        "  - calcula os gradientes a partir de cada .grad_fn;\n"
        "  - acumula esses gradientes no atributo .grad do respectivo tensor;\n"
        "  - propaga tudo até as folhas, seguindo a regra da cadeia."
    )


def jacobian_product_demo() -> None:
    """Quando a saída não é escalar, .backward() precisa de um tensor da
    mesma shape (aqui, torch.ones_like(out)) para calcular o produto
    Jacobiano-vetor. Como os gradientes se acumulam em .grad a cada chamada,
    é preciso zerar (.grad.zero_()) entre chamadas para não somar com a
    anterior."""
    inp = torch.eye(4, 5, requires_grad=True)
    out = (inp + 1).pow(2).t()

    out.backward(torch.ones_like(out), retain_graph=True)
    print(f"Primeira chamada:\n{inp.grad}")

    out.backward(torch.ones_like(out), retain_graph=True)
    print(f"\nSegunda chamada (gradiente acumulado):\n{inp.grad}")

    inp.grad.zero_()
    out.backward(torch.ones_like(out), retain_graph=True)
    print(f"\nChamada após zerar o gradiente:\n{inp.grad}")


def _print_section(title: str) -> None:
    separator = "-" * 60
    print(f"\n{separator}\n{title}\n{separator}")


if __name__ == "__main__":
    _print_section("Grafo computacional")
    x, w, b, z, loss = build_loss_graph()
    print_grad_functions(z, loss)

    _print_section("Calculando gradientes")
    compute_gradients(loss, w, b)

    _print_section("Desabilitando o rastreamento de gradiente")
    demo_disable_gradient_tracking(x, w, b)

    _print_section("Mais sobre o grafo computacional")
    print_computational_graph_explanation()

    _print_section("Gradientes de tensores e produto Jacobiano")
    jacobian_product_demo()
