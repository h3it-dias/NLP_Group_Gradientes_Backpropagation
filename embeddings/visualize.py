"""Diagrams for the three embedding models: static architecture diagrams
(layers + shapes) and autograd computation graphs, both rendered with
graphviz. The computation graphs use torchviz.make_dot, the PyTorch
equivalent of micrograd's draw_dot — nodes are operations from the backward
graph, but since these are real tensors (not micrograd scalars) each node
shows an operation name and tensor shape rather than a single data/grad
number."""
from __future__ import annotations

from pathlib import Path

import graphviz
import torch
from torchviz import make_dot

from .cbow import CBOWModel
from .simple_embedding import EmbeddingBagClassifier
from .skipgram import SkipGramModel

OUTPUT_DIR = Path(__file__).parent / "diagrams"


def build_architecture_graph(title: str, stages: list[str]) -> graphviz.Digraph:
    """Lay out `stages` (already-formatted node labels) top-to-bottom as a
    single sequential pipeline."""
    graph = graphviz.Digraph(name=title.replace(" ", "_"))
    graph.attr(label=title, labelloc="t", fontsize="16")
    graph.attr("node", shape="box", style="rounded,filled", fillcolor="#eef2ff", fontname="Helvetica")
    graph.attr("edge", fontname="Helvetica", fontsize="10")

    for i, stage in enumerate(stages):
        graph.node(str(i), stage)
        if i > 0:
            graph.edge(str(i - 1), str(i))

    return graph


def simple_embedding_graph(vocab_size: int = "V", embedding_dim: int = "D", num_classes: int = "C") -> graphviz.Digraph:
    return build_architecture_graph(
        "EmbeddingBagClassifier (simple_embedding.py)",
        [
            "Entrada\ntokens (N,) + offsets (batch,)",
            f"nn.EmbeddingBag\n{vocab_size} → {embedding_dim}\nmode=mean",
            f"nn.Linear\n{embedding_dim} → {num_classes}",
            f"Saída\nlogits (batch, {num_classes})",
        ],
    )


def cbow_graph(vocab_size: int = "V", embedding_dim: int = "D", window: int = "W") -> graphviz.Digraph:
    return build_architecture_graph(
        "CBOWModel (cbow.py)",
        [
            f"Entrada\ncontexto (batch, 2*{window})",
            f"nn.Embedding\n{vocab_size} → {embedding_dim}",
            f"mean(dim=1)\n(batch, {embedding_dim})",
            f"nn.Linear\n{embedding_dim} → {vocab_size}",
            f"Saída\nlogits (batch, {vocab_size})",
        ],
    )


def skipgram_graph(vocab_size: int = "V", embedding_dim: int = "D") -> graphviz.Digraph:
    return build_architecture_graph(
        "SkipGramModel (skipgram.py)",
        [
            "Entrada\ncentro (batch,)",
            f"nn.Embedding\n{vocab_size} → {embedding_dim}",
            f"nn.Linear\n{embedding_dim} → {vocab_size}",
            f"Saída\nlogits (batch, {vocab_size})",
        ],
    )


def _styled_autograd_graph(dot: graphviz.Digraph, title: str) -> graphviz.Digraph:
    dot.attr(rankdir="LR", label=title, labelloc="t", fontsize="16")
    return dot


def simple_embedding_autograd_graph(
    vocab_size: int = 10, embedding_dim: int = 4, num_classes: int = 2
) -> graphviz.Digraph:
    model = EmbeddingBagClassifier(vocab_size, embedding_dim, num_classes)
    tokens = torch.tensor([1, 4, 7])
    offsets = torch.tensor([0])
    output = model(tokens, offsets)
    dot = make_dot(output, params=dict(model.named_parameters()))
    return _styled_autograd_graph(dot, "EmbeddingBagClassifier — grafo computacional")


def cbow_autograd_graph(vocab_size: int = 10, embedding_dim: int = 4, window: int = 2) -> graphviz.Digraph:
    model = CBOWModel(vocab_size, embedding_dim)
    context = torch.randint(0, vocab_size, (1, 2 * window))
    output = model(context)
    dot = make_dot(output, params=dict(model.named_parameters()))
    return _styled_autograd_graph(dot, "CBOWModel — grafo computacional")


def skipgram_autograd_graph(vocab_size: int = 10, embedding_dim: int = 4) -> graphviz.Digraph:
    model = SkipGramModel(vocab_size, embedding_dim)
    center = torch.tensor([3])
    output = model(center)
    dot = make_dot(output, params=dict(model.named_parameters()))
    return _styled_autograd_graph(dot, "SkipGramModel — grafo computacional")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    graphs = {
        "simple_embedding": simple_embedding_graph(),
        "cbow": cbow_graph(),
        "skipgram": skipgram_graph(),
        "simple_embedding_autograd": simple_embedding_autograd_graph(),
        "cbow_autograd": cbow_autograd_graph(),
        "skipgram_autograd": skipgram_autograd_graph(),
    }
    for name, graph in graphs.items():
        path = graph.render(directory=OUTPUT_DIR, filename=name, format="png", cleanup=True)
        print(f"salvo em {path}")
