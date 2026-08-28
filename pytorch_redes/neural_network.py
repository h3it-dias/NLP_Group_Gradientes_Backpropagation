"""Modelo base de rede neural para classificação de imagens, e exploração das
camadas usadas para construí-lo (nn.Flatten, nn.Linear, nn.ReLU, nn.Sequential)."""
from __future__ import annotations

import torch
from torch import nn

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


class NeuralNetwork(nn.Module):  # Uso do nn.Module para criar qualquer rede
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def predict(model: NeuralNetwork, x: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    pred_probab = nn.Softmax(dim=1)(logits)
    return pred_probab.argmax(1)


def explore_layers(input_image: torch.Tensor) -> None:
    """Demonstra, passo a passo, o que cada camada faz com um tensor de imagem."""
    print(f"Tamanho da imagem de entrada: {input_image.size()}")

    flatten = nn.Flatten()
    flat_image = flatten(input_image)
    print(f"Depois do Flatten: {flat_image.size()}")

    layer1 = nn.Linear(in_features=28 * 28, out_features=20)
    hidden1 = layer1(flat_image)
    print(f"Depois do Linear: {hidden1.size()}")
    print(f"Antes do ReLU: {hidden1}\n")
    hidden1 = nn.ReLU()(hidden1)
    print(f"Depois do ReLU: {hidden1}")

    seq_modules = nn.Sequential(flatten, layer1, nn.ReLU(), nn.Linear(20, 10))
    logits = seq_modules(input_image)
    pred_probab = nn.Softmax(dim=1)(logits)
    print(f"Probabilidades previstas: {pred_probab}")


def print_parameters(model: NeuralNetwork) -> None:
    for name, param in model.named_parameters():
        print(f"Camada: {name} | Tamanho: {param.size()} | Valores: {param[:2]}\n")


def _print_section(title: str) -> None:
    separator = "-" * 60
    print(f"\n{separator}\n{title}\n{separator}")


if __name__ == "__main__":
    _print_section("Dispositivo")
    print(f"Using {DEVICE} device")

    _print_section("Modelo")
    model = NeuralNetwork().to(DEVICE)
    print(model)

    _print_section("Predição de exemplo")
    x = torch.rand(1, 28, 28, device=DEVICE)
    y_pred = predict(model, x)
    print(f"Predicted class: {y_pred}")

    _print_section("Exploração de camadas")
    explore_layers(torch.rand(3, 28, 28))

    _print_section("Parâmetros do modelo")
    print(f"Model structure: {model}\n")
    print_parameters(model)
