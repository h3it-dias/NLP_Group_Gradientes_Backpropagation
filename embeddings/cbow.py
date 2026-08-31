"""Word2Vec CBOW: predicts the center word from its surrounding context words."""
from __future__ import annotations

import torch
from torch import nn

from .vocab import TOY_CORPUS, Vocab, build_vocab, nearest_neighbors, tokenize


class CBOWModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        # context: (batch, 2 * window) -> average the context embeddings
        embedded = self.embeddings(context).mean(dim=1)
        return self.linear(embedded)


def make_cbow_pairs(corpus: list[str], vocab: Vocab, window: int = 2) -> list[tuple[list[int], int]]:
    pairs = []
    for line in corpus:
        tokens = [vocab.encode(tok) for tok in tokenize(line)]
        for i, center in enumerate(tokens):
            context = tokens[max(0, i - window):i] + tokens[i + 1:i + window + 1]
            if len(context) == 2 * window:
                pairs.append((context, center))
    return pairs


def train_demo(
    epochs: int = 300, embedding_dim: int = 16, window: int = 2, lr: float = 0.05
) -> tuple[CBOWModel, Vocab]:
    vocab = build_vocab(TOY_CORPUS)
    pairs = make_cbow_pairs(TOY_CORPUS, vocab, window=window)
    contexts = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    centers = torch.tensor([p[1] for p in pairs], dtype=torch.long)

    model = CBOWModel(len(vocab), embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(contexts)
        loss = loss_fn(logits, centers)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0 or epoch == 1:
            print(f"epoch {epoch:>3d} | loss {loss.item():.4f}")

    return model, vocab


if __name__ == "__main__":
    model, vocab = train_demo()
    for word in ["gato", "cachorro"]:
        neighbors = nearest_neighbors(word, vocab, model.embeddings.weight.detach())
        print(f"vizinhos de '{word}': {neighbors}")
