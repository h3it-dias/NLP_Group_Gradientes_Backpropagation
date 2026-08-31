"""Vocabulary and toy corpus shared by the embedding models in this package."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import torch

UNK_TOKEN = "<unk>"

TOY_CORPUS = [
    "o gato preto dorme no sofa",
    "o cachorro marrom corre no parque",
    "o gato branco brinca com o cachorro",
    "o cachorro late para o gato",
    "o gato mia para o cachorro",
    "a crianca brinca com o cachorro no parque",
    "a crianca brinca com o gato no sofa",
]


def tokenize(text: str) -> list[str]:
    return text.lower().split()


class Vocab:
    def __init__(self, tokens: Iterable[str]):
        counts = Counter(tokens)
        self.itos = [UNK_TOKEN] + sorted(counts, key=lambda tok: (-counts[tok], tok))
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, token: str) -> int:
        return self.stoi.get(token, self.stoi[UNK_TOKEN])

    def decode(self, index: int) -> str:
        return self.itos[index]


def build_vocab(corpus: Sequence[str]) -> Vocab:
    tokens = [tok for line in corpus for tok in tokenize(line)]
    return Vocab(tokens)


def nearest_neighbors(
    word: str, vocab: Vocab, weights: torch.Tensor, top_k: int = 5
) -> list[tuple[str, float]]:
    """Rank vocabulary words by cosine similarity to `word`'s embedding."""
    idx = vocab.encode(word)
    target = weights[idx]
    sims = torch.nn.functional.cosine_similarity(weights, target.unsqueeze(0))
    top = torch.topk(sims, min(top_k + 1, len(vocab)))

    results = []
    for score, i in zip(top.values.tolist(), top.indices.tolist()):
        if i == idx:
            continue
        results.append((vocab.decode(i), score))
        if len(results) == top_k:
            break
    return results
