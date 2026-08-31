"""Visualização do grafo computacional montado pela Value. Mesmo espírito do
torchviz.make_dot usado em pytorch_redes/embeddings/visualize.py, mas aqui
cada nó já é literalmente um escalar, então dá pra mostrar data e grad de
verdade em cada um, em vez de só o shape do tensor."""
from __future__ import annotations

from graphviz import Digraph

from .engine import Value


def trace(root: Value) -> tuple[set[Value], set[tuple[Value, Value]]]:
    """Monta o conjunto de nós e arestas do grafo alcançável a partir de root."""
    nodes: set[Value] = set()
    edges: set[tuple[Value, Value]] = set()

    def build(v: Value) -> None:
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root: Value) -> Digraph:
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # para cada valor do grafo, cria um nó retangular ('record') com data e grad
        dot.node(name=uid, label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape="record")
        if n._op:
            # se esse valor é resultado de uma operação, cria um nó pra ela e conecta
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # conecta n1 ao nó de operação de n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
