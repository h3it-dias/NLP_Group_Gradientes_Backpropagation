"""A menor rede possível com uma camada de embedding: uma palavra -> um vetor
-> uma camada linear -> logits -> softmax -> loss. O objetivo aqui não é
aprender embeddings úteis (isso já é feito em cbow.py/skipgram.py), é isolar
um único passo de treino pra observar o que .backward() faz com .grad, sem
nenhum ruído de otimizador, batch ou pooling no meio.

O forward é feito escalar por escalar (cada multiplicação peso*entrada é sua
própria operação) em vez de chamar model.linear(embedded) de uma vez só —
exatamente como o Neuron do micrograd faz, mas com tensores reais do
PyTorch. Isso é mais lento e nada idiomático pra uso real, mas é o que
permite desenhar cada peso e cada operação como um nó separado no grafo
(ver embeddings/visualize.py:gradient_step_graph)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .vocab import TOY_CORPUS, Vocab, build_vocab


@dataclass
class LearningStepResult:
    """Tudo que a visualização (embeddings/visualize.py) precisa pra desenhar
    o grafo com data/grad reais em cada nó, no estilo do micrograd."""

    model: "MinimalEmbeddingModel"
    vocab: Vocab
    word: str
    word_id: torch.Tensor
    target_class: int
    embedding_dim: int
    num_classes: int
    neuron_terms: list[list[torch.Tensor]]  # neuron_terms[j][i] = embedding[i] * weight[j][i]
    logits: list[torch.Tensor]  # logits[j], escalar
    probs: torch.Tensor  # (num_classes,), softmax dos logits
    loss: torch.Tensor  # escalar


class MinimalEmbeddingModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, token_id: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_id)  # (batch, embedding_dim)
        return self.linear(embedded)  # (batch, num_classes)


def print_grad(model: MinimalEmbeddingModel, word_id: torch.Tensor, label: str) -> None:
    """Mostra o .grad de dois parâmetros: a linha da tabela de embedding
    correspondente à palavra usada, e a matriz de pesos da camada linear."""
    embedding_grad = model.embedding.weight.grad
    linear_grad = model.linear.weight.grad

    print(f"[{label}]")
    if embedding_grad is None:
        print("  embedding.weight.grad: None (backward ainda não rodou)")
    else:
        print(f"  embedding.weight.grad (linha da palavra): {embedding_grad[word_id.item()]}")
    if linear_grad is None:
        print("  linear.weight.grad: None (backward ainda não rodou)")
    else:
        print(f"  linear.weight.grad:\n{linear_grad}")


def one_learning_step(
    word: str = "gato", target_class: int = 1, embedding_dim: int = 4, num_classes: int = 2
) -> LearningStepResult:
    """Forward -> loss -> backward, um único passo, sem otimizador."""
    vocab = build_vocab(TOY_CORPUS)
    torch.manual_seed(0)
    model = MinimalEmbeddingModel(vocab_size=len(vocab), embedding_dim=embedding_dim, num_classes=num_classes)

    word_id = torch.tensor([vocab.encode(word)])

    print_grad(model, word_id, "Antes do backward")

    embedding_row = model.embedding.weight[word_id.item()]

    # cada neurônio de saída (uma classe) é: soma(peso_i * entrada_i) + bias,
    # exatamente como o Neuron do micrograd — só que aqui cada termo
    # peso_i*entrada_i é seu próprio tensor com retain_grad(), pra poder
    # mostrar o gradiente individual de cada multiplicação no diagrama.
    neuron_terms: list[list[torch.Tensor]] = []
    logits: list[torch.Tensor] = []
    for j in range(num_classes):
        terms_j = [embedding_row[i] * model.linear.weight[j, i] for i in range(embedding_dim)]
        for term in terms_j:
            term.retain_grad()

        logit = terms_j[0]
        for term in terms_j[1:]:
            logit = logit + term
        logit = logit + model.linear.bias[j]
        logit.retain_grad()

        neuron_terms.append(terms_j)
        logits.append(logit)

    probs = torch.softmax(torch.stack(logits), dim=0)
    probs.retain_grad()
    loss = -torch.log(probs[target_class])
    loss.retain_grad()

    print(f"\nlogits: {[round(l.item(), 4) for l in logits]}")
    print(f"probs (softmax): {[round(p, 4) for p in probs.detach().tolist()]}")
    print(f"loss: {loss.item():.4f}\n")

    loss.backward()

    print_grad(model, word_id, "Depois do backward")

    return LearningStepResult(
        model=model,
        vocab=vocab,
        word=word,
        word_id=word_id,
        target_class=target_class,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        neuron_terms=neuron_terms,
        logits=logits,
        probs=probs,
        loss=loss,
    )


def intervention_demo(
    word: str = "gato", baseline_target: int = 1, new_target: int = 0
) -> tuple[LearningStepResult, LearningStepResult]:
    """Roda o mesmo passo de aprendizado duas vezes — mudando só o target —
    pra isolar o efeito dessa única mudança sobre o gradiente. Como
    one_learning_step fixa a semente aleatória (torch.manual_seed(0)) antes
    de criar o modelo, as duas rodadas partem exatamente dos mesmos pesos;
    qualquer diferença nos gradientes vem só da mudança de target."""
    print(f"########## BASELINE (target_class={baseline_target}) ##########")
    baseline = one_learning_step(word=word, target_class=baseline_target)

    print(f"\n########## INTERVENÇÃO (target_class={new_target}) ##########")
    intervened = one_learning_step(word=word, target_class=new_target)

    print("\n########## COMPARAÇÃO ##########")
    print(f"probs (baseline):    {[round(p, 4) for p in baseline.probs.detach().tolist()]}")
    print(f"probs (intervenção): {[round(p, 4) for p in intervened.probs.detach().tolist()]}")
    print(f"loss (baseline):     {baseline.loss.item():.4f}")
    print(f"loss (intervenção):  {intervened.loss.item():.4f}\n")

    print(f"{'':22s}{'baseline':>12s}{'intervenção':>14s}")
    base_emb_grad = baseline.model.embedding.weight.grad[baseline.word_id.item()]
    interv_emb_grad = intervened.model.embedding.weight.grad[intervened.word_id.item()]
    for i in range(baseline.embedding_dim):
        print(f"embedding.grad[{i}]{'':6s}{base_emb_grad[i].item():>12.4f}{interv_emb_grad[i].item():>14.4f}")

    for j in range(baseline.num_classes):
        for i in range(baseline.embedding_dim):
            g1 = baseline.model.linear.weight.grad[j, i].item()
            g2 = intervened.model.linear.weight.grad[j, i].item()
            print(f"w[{j}][{i}].grad{'':7s}{g1:>12.4f}{g2:>14.4f}")

    return baseline, intervened


if __name__ == "__main__":
    intervention_demo()
