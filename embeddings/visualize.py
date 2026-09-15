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
from .gradient_step import LearningStepResult, intervention_demo
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


LEAF_COLOR = "#eef2ff"  # parâmetros treináveis (folhas do grafo)
ROOT_COLOR = "#d7f5d7"  # loss: raiz do .backward(), onde grad = 1.0 nasce
GRAD_NOT_COMPUTED = "ainda não calculado"


def _scalar(t: torch.Tensor, decimals: int = 4) -> str:
    return f"{t.item():.{decimals}f}"


def gradient_step_graph(result: LearningStepResult, stage: str = "depois") -> graphviz.Digraph:
    """Grafo computacional do passo mínimo de aprendizado (embeddings/
    gradient_step.py), desenrolado escalar por escalar — cada peso, cada
    componente do embedding e cada multiplicação vira seu próprio nó, no
    mesmo estilo do draw_dot do micrograd (uma caixa por Value).

    stage="antes" desenha o grafo logo após o forward, mas mostrando grad
    como "ainda não calculado" em vez do valor real — porque antes do
    .backward() rodar, .grad de fato é None em todo mundo. stage="depois"
    mostra os valores reais de grad, já calculados pela regra da cadeia.
    """
    show_grad = stage != "antes"
    model = result.model
    d = result.embedding_dim
    c = result.num_classes
    embedding_row = model.embedding.weight[result.word_id.item()]
    embedding_row_grad = model.embedding.weight.grad[result.word_id.item()] if model.embedding.weight.grad is not None else None

    def grad_text(t: torch.Tensor | None) -> str:
        return _scalar(t) if show_grad and t is not None else GRAD_NOT_COMPUTED

    dot = graphviz.Digraph(format="svg", graph_attr={"rankdir": "LR"})
    title = {
        "antes": "Passo mínimo de aprendizado — ANTES do backward (gradient_step.py)",
        "depois": "Passo mínimo de aprendizado — DEPOIS do backward (gradient_step.py)",
        "intervencao": f"Intervenção — target_class={result.target_class} (gradient_step.py)",
    }[stage]
    dot.attr(
        label=title,
        labelloc="t",
        fontsize="16",
        nodesep="0.35",
        ranksep="1.1",
        splines="polyline",
    )
    dot.attr("node", fontname="Helvetica", fontsize="11")
    dot.attr("edge", fontname="Helvetica", fontsize="10")

    def record(node_id: str, label: str, value_text: str, grad_value: torch.Tensor | None, fillcolor: str | None = None) -> None:
        # rótulo HTML-like em vez de shape="record": nós record não aceitam
        # arestas invisíveis de ordenação no mesmo rank (graphviz recusa a
        # renderizar - "flat edge between adjacent nodes... record shape").
        bgcolor = fillcolor or "white"
        html = (
            '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4" BGCOLOR="%s">'
            "<TR><TD>%s</TD><TD>data %s</TD><TD>grad %s</TD></TR>"
            "</TABLE>>"
        ) % (bgcolor, label, value_text, grad_text(grad_value))
        dot.node(node_id, html, shape="none")

    def order_top_to_bottom(node_ids: list[str], graph: graphviz.Digraph = dot) -> None:
        """Força a ordem vertical de nós que o graphviz colocaria na mesma
        coluna (mesmo rank) — sem isso, o dot escolhe a ordem sozinho pra
        minimizar cruzamento de linhas, o que embaralha o índice.

        Quando os nós pertencem a um cluster, esse rank=same PRECISA ser
        declarado dentro do subgraph do próprio cluster (passe `graph=cluster`)
        — se for um subgraph irmão referenciando os mesmos nós, o dot
        descarta o cluster inteiro sem avisar."""
        with graph.subgraph() as s:
            s.attr(rank="same")
            for node_id in node_ids:
                s.node(node_id)
            for a, b in zip(node_ids, node_ids[1:]):
                s.edge(a, b, style="invis")

    leaf_order: list[str] = []

    # e0, e1, e2, ... primeiro, em ordem — são compartilhados pelos dois
    # neurônios, então ficam num bloco à parte no topo.
    for i in range(d):
        e_i = embedding_row[i]
        e_i_grad = embedding_row_grad[i] if embedding_row_grad is not None else None
        record(f"e{i}", f"e{i} (embedding['{result.word}'][{i}])", _scalar(e_i), e_i_grad, LEAF_COLOR)
        leaf_order.append(f"e{i}")

    # pesos agrupados por neurônio (não por índice): assim o bloco de pesos
    # do neurônio j fica alinhado com o bloco de multiplicações/somas desse
    # mesmo neurônio, evitando que as linhas da soma cruzem pra outro bloco.
    for j in range(c):
        for i in range(d):
            leaf_order.append(f"w{j}_{i}")
        leaf_order.append(f"bias{j}")

    for j in range(c):
        for i in range(d):
            w_ji = model.linear.weight[j, i]
            w_ji_grad = model.linear.weight.grad[j, i] if model.linear.weight.grad is not None else None
            record(f"w{j}_{i}", f"w[{j}][{i}]", _scalar(w_ji), w_ji_grad, LEAF_COLOR)

            term = result.neuron_terms[j][i]
            record(f"term{j}_{i}", f"e{i}*w[{j}][{i}]", _scalar(term), term.grad)

            dot.node(f"mul{j}_{i}", "*")
            dot.edge(f"e{i}", f"mul{j}_{i}")
            dot.edge(f"w{j}_{i}", f"mul{j}_{i}")
            dot.edge(f"mul{j}_{i}", f"term{j}_{i}")

        bias_j = model.linear.bias[j]
        bias_j_grad = model.linear.bias.grad[j] if model.linear.bias.grad is not None else None
        record(f"bias{j}", f"linear.bias[{j}]", _scalar(bias_j), bias_j_grad, LEAF_COLOR)

        logit_j = result.logits[j]
        record(f"logit{j}", f"logit[{j}]", _scalar(logit_j), logit_j.grad)

        dot.node(f"sum{j}", "+")
        for i in range(d):
            dot.edge(f"term{j}_{i}", f"sum{j}")
        dot.edge(f"bias{j}", f"sum{j}")
        dot.edge(f"sum{j}", f"logit{j}")

        # agrupa visualmente as multiplicações + termos + soma desse
        # neurônio numa borda só (cluster do graphviz — sai como um
        # retângulo arredondado, não um círculo perfeito, mas cumpre o
        # mesmo papel de "isso tudo é o neurônio j").
        with dot.subgraph(name=f"cluster_neuron{j}") as cluster:
            cluster.attr(label=f"Neurônio {j} (classe {j})", style="rounded", color="gray50", fontsize="12")
            for i in range(d):
                cluster.node(f"mul{j}_{i}")
                cluster.node(f"term{j}_{i}")
            cluster.node(f"sum{j}")

            order_top_to_bottom([f"mul{j}_{i}" for i in range(d)], graph=cluster)
            order_top_to_bottom([f"term{j}_{i}" for i in range(d)], graph=cluster)

    # softmax: todos os logits juntos viram as probabilidades
    dot.node("op_softmax", "softmax")
    for j in range(c):
        dot.edge(f"logit{j}", "op_softmax")

    for j in range(c):
        prob_j = result.probs[j]
        prob_j_grad = result.probs.grad[j] if result.probs.grad is not None else None
        fill = ROOT_COLOR if j == result.target_class else None
        record(f"prob{j}", f"prob[{j}]{' (classe alvo)' if j == result.target_class else ''}", _scalar(prob_j), prob_j_grad, fill)
        dot.edge("op_softmax", f"prob{j}")

    # loss = -log(prob da classe alvo)
    dot.node("op_nll", "-log")
    dot.edge(f"prob{result.target_class}", "op_nll")
    record("loss", "loss", _scalar(result.loss), result.loss.grad, ROOT_COLOR)
    dot.edge("op_nll", "loss")

    order_top_to_bottom(leaf_order)
    order_top_to_bottom([f"sum{j}" for j in range(c)])
    order_top_to_bottom([f"logit{j}" for j in range(c)])
    order_top_to_bottom([f"prob{j}" for j in range(c)])

    return dot


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    baseline_result, intervened_result = intervention_demo()
    graphs = {
        "simple_embedding": simple_embedding_graph(),
        "cbow": cbow_graph(),
        "skipgram": skipgram_graph(),
        "simple_embedding_autograd": simple_embedding_autograd_graph(),
        "cbow_autograd": cbow_autograd_graph(),
        "skipgram_autograd": skipgram_autograd_graph(),
        "gradient_step_antes": gradient_step_graph(baseline_result, stage="antes"),
        "gradient_step_depois": gradient_step_graph(baseline_result, stage="depois"),
        "gradient_step_intervencao": gradient_step_graph(intervened_result, stage="intervencao"),
    }
    for name, graph in graphs.items():
        path = graph.render(directory=OUTPUT_DIR, filename=name, format="png", cleanup=True)
        print(f"salvo em {path}")
