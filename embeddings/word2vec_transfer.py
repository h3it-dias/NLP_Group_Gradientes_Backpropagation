"""Exemplo prático do uso de embeddings estilo word2vec: treina um CBOW no
mesmo corpus do classificador de sentimento (simple_embedding.py) e usa a
tabela de embeddings resultante para inicializar o classificador, em vez de
pesos aleatórios. É a mesma ideia por trás de usar embeddings pré-treinados
(word2vec, GloVe, ...) numa tarefa de classificação de texto: o embedding
já chega sabendo algo sobre como as palavras se relacionam, em vez de
começar do zero.

As frases de sentimento são curtas (3-5 palavras), então o CBOW aqui usa
window=1 — com o window=2 padrão de cbow.py quase nenhuma frase teria
contexto suficiente pra gerar um par de treino."""
from __future__ import annotations

import torch

from .cbow import train_demo as train_cbow
from .simple_embedding import NEGATIVE, POSITIVE, train_demo as train_classifier
from .vocab import Vocab


def pretrain_embeddings(embedding_dim: int = 16) -> tuple[torch.Tensor, Vocab]:
    """Treina CBOW no corpus de sentimento (não no TOY_CORPUS de gato/
    cachorro) e devolve a tabela de embeddings resultante, já alinhada ao
    vocabulário desse corpus."""
    sentences = POSITIVE + NEGATIVE
    cbow_model, vocab = train_cbow(corpus=sentences, embedding_dim=embedding_dim, window=1)
    return cbow_model.embeddings.weight.detach().clone(), vocab


def compare_demo(epochs: int = 200, embedding_dim: int = 16) -> None:
    print("########## Pré-treinando embeddings com CBOW no corpus de sentimento ##########")
    pretrained_weights, vocab = pretrain_embeddings(embedding_dim=embedding_dim)

    print("\n########## Classificador com embedding aleatório (baseline) ##########")
    torch.manual_seed(0)
    _, _, losses_random = train_classifier(epochs=epochs, embedding_dim=embedding_dim, vocab=vocab, verbose=False)

    print("########## Classificador com embedding pré-treinado (CBOW) ##########")
    torch.manual_seed(0)
    _, _, losses_pretrained = train_classifier(
        epochs=epochs, embedding_dim=embedding_dim, vocab=vocab, pretrained_embedding=pretrained_weights, verbose=False
    )

    print("\n########## Comparação da perda ao longo do treino ##########")
    print(f"{'epoch':>6s}{'aleatório':>14s}{'pré-treinado':>16s}")
    checkpoints = sorted({1, 5, 10, 25, 50, 100, epochs})
    for epoch in checkpoints:
        idx = epoch - 1
        print(f"{epoch:>6d}{losses_random[idx]:>14.4f}{losses_pretrained[idx]:>16.4f}")


if __name__ == "__main__":
    compare_demo()
