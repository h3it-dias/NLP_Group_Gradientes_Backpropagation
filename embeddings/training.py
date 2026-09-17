"""Loop de treino compartilhado por cbow.py, skipgram.py e simple_embedding.py
— o mecanismo (forward -> loss -> backward -> step -> log) é idêntico nos
três; o que muda entre eles é só o modelo, os dados e o otimizador."""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


def train_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    forward_fn: Callable[[], torch.Tensor],
    targets: torch.Tensor,
    epochs: int = 200,
    loss_fn: nn.Module | None = None,
    verbose: bool = True,
    log_every: int = 50,
) -> list[float]:
    """`forward_fn` é uma função sem argumentos que já sabe como chamar o
    modelo (fecha sobre os tensores de entrada de cada caso — tokens/offsets,
    contexts, centers etc.), porque essa parte muda de modelo pra modelo."""
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = forward_fn()
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if verbose and (epoch % log_every == 0 or epoch == 1):
            print(f"epoch {epoch:>3d} | loss {loss.item():.4f}")
    return losses
