"""Simplest use of an embedding layer: nn.EmbeddingBag feeding a linear
classifier, trained on a tiny toy sentiment dataset."""
from __future__ import annotations

import torch
from torch import nn

from .training import train_loop
from .vocab import Vocab, build_vocab, tokenize

POSITIVE = [
    "eu amo esse filme",
    "o dia esta lindo hoje",
    "que comida deliciosa",
    "estou muito feliz",
    "isso foi otimo",
]
NEGATIVE = [
    "eu odeio esse filme",
    "o dia esta horrivel hoje",
    "que comida horrivel",
    "estou muito triste",
    "isso foi pessimo",
]


class EmbeddingBagClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, num_classes: int):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, mode="mean")
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, tokens: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens, offsets)
        return self.classifier(embedded)


def encode_dataset(sentences: list[str], vocab: Vocab) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten sentences into one token tensor plus per-sentence start offsets,
    the layout nn.EmbeddingBag expects."""
    tokens, offsets = [], [0]
    for sentence in sentences:
        ids = [vocab.encode(tok) for tok in tokenize(sentence)]
        tokens.extend(ids)
        offsets.append(offsets[-1] + len(ids))
    return torch.tensor(tokens, dtype=torch.long), torch.tensor(offsets[:-1], dtype=torch.long)


def train_demo(
    epochs: int = 200,
    embedding_dim: int = 16,
    lr: float = 0.1,
    vocab: Vocab | None = None,
    pretrained_embedding: torch.Tensor | None = None,
    verbose: bool = True,
) -> tuple[EmbeddingBagClassifier, Vocab, list[float]]:
    """Treina o classificador de sentimento. Se `pretrained_embedding` for
    passado (uma tabela vocab_size x embedding_dim já treinada em outra
    tarefa, ex. CBOW), ela substitui a inicialização aleatória do
    nn.EmbeddingBag — é o "aproveitar embeddings prontos" que word2vec/GloVe
    fazem na prática. `vocab` precisa ser o vocabulário usado pra treinar
    esse embedding, senão os índices das palavras não batem."""
    sentences = POSITIVE + NEGATIVE
    labels = torch.tensor([1] * len(POSITIVE) + [0] * len(NEGATIVE), dtype=torch.long)
    if vocab is None:
        vocab = build_vocab(sentences)

    tokens, offsets = encode_dataset(sentences, vocab)
    model = EmbeddingBagClassifier(len(vocab), embedding_dim, num_classes=2)
    if pretrained_embedding is not None:
        model.embedding.weight = nn.Parameter(pretrained_embedding.clone())

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses = train_loop(model, optimizer, lambda: model(tokens, offsets), labels, epochs=epochs, verbose=verbose)

    return model, vocab, losses


if __name__ == "__main__":
    model, vocab, _ = train_demo()

    test_sentences = ["eu amo esse dia", "isso foi horrivel"]
    tokens, offsets = encode_dataset(test_sentences, vocab)
    model.eval()
    with torch.no_grad():
        preds = model(tokens, offsets).argmax(dim=1)
    for sentence, pred in zip(test_sentences, preds):
        label = "positivo" if pred.item() == 1 else "negativo"
        print(f'"{sentence}" -> {label}')
