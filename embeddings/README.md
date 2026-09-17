# embeddings

Projeto prático do grupo: usa embeddings de palavras como fio condutor pra
explicar gradiente, regra da cadeia e `.backward()` na prática, e depois
mostra uma aplicação real desses embeddings.

> **Nota:** para a entrega "passo mínimo + intervenção" (Partes 1 e 2 abaixo,
> em `gradient_step.py`), a versão de apresentação do grupo passou a ser
> `notebooks/backprop_do_zero.ipynb` (motor `Value`, sem precisar decompor
> `nn.Linear` escalar por escalar). `gradient_step.py` continua funcional e
> fica aqui como referência/material de estudo. A Parte 3
> (`word2vec_transfer.py`, embeddings pré-treinados) não tem equivalente lá
> e continua sendo a peça de referência para esse tema.

## Estrutura

**Fundamentos (vocabulário e três arquiteturas de embedding):**
- `vocab.py` — vocabulário compartilhado (`Vocab`, `build_vocab`, `tokenize`), `TOY_CORPUS` (frases sobre gato/cachorro) e `nearest_neighbors` (busca por similaridade de cosseno).
- `simple_embedding.py` — `nn.EmbeddingBag` + classificador linear de sentimento (positivo/negativo), treinado num corpus de frases de sentimento (`POSITIVE`/`NEGATIVE`).
- `cbow.py` / `skipgram.py` — Word2Vec CBOW e Skip-gram, treinados por padrão no `TOY_CORPUS`.
- `training.py` — `train_loop`, o loop de treino (forward → loss → backward → step → log) compartilhado por `cbow.py`, `skipgram.py` e `simple_embedding.py`.
- `visualize.py` — diagramas de arquitetura (camadas/shapes) e grafos de autograd (via `torchviz.make_dot`) dos três modelos acima.

**Os 3 experimentos didáticos sobre gradiente:**
1. `gradient_step.py` — a menor rede possível (embedding + linear), um único passo forward → softmax → loss → backward, com `.grad` mostrado antes e depois.
2. `gradient_step.py:intervention_demo` — o mesmo passo repetido mudando só o `target`, comparando como os gradientes mudam.
3. `word2vec_transfer.py` — pré-treina embeddings com CBOW no corpus de sentimento e os usa pra inicializar o classificador do `simple_embedding.py`, comparando com inicialização aleatória.

Os diagramas de todos os módulos vão para `embeddings/diagrams/` (gerados, não versionados).

## Como executar

Com o `.venv` ativado, a partir da raiz do repositório:

```bash
python3 -m embeddings.simple_embedding
python3 -m embeddings.cbow
python3 -m embeddings.skipgram
python3 -m embeddings.gradient_step
python3 -m embeddings.word2vec_transfer
python3 -m embeddings.visualize   # gera todos os diagramas de uma vez
```

Dependências: `torch`, `torchvision`, `graphviz` (+ binário `dot` do sistema), `torchviz`.

> Nenhum dos três modelos abaixo fixa `torch.manual_seed`, então os números
> exatos de perda e vizinhos variam a cada execução — o padrão (perda caindo,
> vizinhos fazendo algum sentido) é o que importa, não o valor exato.

---

## Fundamentos — vocabulário e as três arquiteturas de embedding

### `simple_embedding.py`

```
epoch   1 | loss 0.6875
epoch  50 | loss 0.4858
epoch 100 | loss 0.3604
epoch 150 | loss 0.2652
epoch 200 | loss 0.1958
"eu amo esse dia" -> positivo
"isso foi horrivel" -> negativo
```

A perda cai a cada 50 épocas impressa (comportamento esperado de um
classificador aprendendo); as duas frases de teste (que não estavam no
treino) são classificadas corretamente ao final.

### `cbow.py` / `skipgram.py`

```
# cbow.py
epoch   1 | loss 3.0978
epoch 300 | loss 0.0003
vizinhos de 'gato': [('a', 0.294), ('marrom', 0.290), ('parque', 0.237), ...]
vizinhos de 'cachorro': [('mia', 0.490), ('<unk>', 0.368), ('late', 0.319), ...]
```

A perda do CBOW despenca a quase zero — é overfitting normal num corpus de
7 frases (`TOY_CORPUS`), não um sinal de que os embeddings ficaram bons.
`vizinhos de 'X'` são as palavras cujo vetor de embedding tem maior
similaridade de cosseno com o de `X` (função `nearest_neighbors` em
`vocab.py`) — com um corpus tão pequeno, os vizinhos costumam ser bem
ruidosos (não espere sinônimos de verdade). O `skipgram.py` imprime o
mesmo formato; a perda estabiliza mais alto (~1.7) em vez de ir a zero,
porque cada centro tem que prever *várias* palavras de contexto diferentes,
o que dificulta decorar o corpus.

### Diagramas de `visualize.py`

Dois tipos, para os três modelos acima:

**Arquitetura** (`simple_embedding.png`, `cbow.png`, `skipgram.png`) — um
fluxograma simples, de cima pra baixo, com uma caixa por camada mostrando
o nome (`nn.EmbeddingBag`, `nn.Linear`, ...) e o shape que entra/sai
(`V`=tamanho do vocabulário, `D`=dimensão do embedding, `C`=número de
classes, `W`=tamanho da janela de contexto). Não roda nada de verdade —
é só uma descrição estática da arquitetura.

**Grafo de autograd** (`simple_embedding_autograd.png`, etc.) — gerado de
verdade com `torchviz.make_dot` a partir de um forward real. Azul =
parâmetro treinável (nome + shape, ex. `embedding.weight (10, 4)`); cinza
= uma operação do grafo de `.backward()` (o nome é literalmente a classe
interna do PyTorch, ex. `AccumulateGrad` é o nó que acumula gradiente num
parâmetro folha, `EmbeddingBagBackward0`/`TBackward0`/`AddmmBackward0` são
as operações de lookup, transposição e `matmul+bias` que compõem o
`nn.EmbeddingBag`/`nn.Linear`); verde = o tensor de saída final. Esse
grafo mostra *shape*, não o valor de cada peso — pra isso é a Parte 1
abaixo.

---

## Parte 1 — passo mínimo de aprendizado

`gradient_step.py` define `MinimalEmbeddingModel`: uma palavra vira um vetor
de embedding (`embedding_dim=4`), que alimenta uma camada linear com 2
neurônios de saída (2 classes). Diferente de um `nn.Linear` normal, o
forward é feito **escalar por escalar** — cada multiplicação peso×entrada é
sua própria operação com `.retain_grad()` — só pra poder desenhar cada uma
individualmente depois.

Rodando `one_learning_step()`:

```
[Antes do backward]
  embedding.weight.grad: None (backward ainda não rodou)
  linear.weight.grad: None (backward ainda não rodou)

logits: [0.2722, 0.132]
probs (softmax): [0.535, 0.465]
loss: 0.7657

[Depois do backward]
  embedding.weight.grad (linha da palavra): tensor([ 0.0579,  0.0440, -0.0561, -0.0195])
  linear.weight.grad:
tensor([[ 0.0641,  0.6622,  0.5975, -0.1323],
        [-0.0641, -0.6622, -0.5975,  0.1323]])
```

`embeddings/diagrams/gradient_step_antes.png` e `gradient_step_depois.png`
mostram o mesmo grafo, lado a lado: antes, todo `grad` aparece como "ainda
não calculado"; depois, os valores reais aparecem. Cada peso, cada
componente do embedding e cada multiplicação tem sua própria caixa (`data`
e `grad`), no estilo do `draw_dot` do micrograd — os dois neurônios de
saída aparecem agrupados visualmente num contorno próprio.

**Cores das caixas:** azul = parâmetro treinável (peso, bias ou componente
do embedding — o que o `optimizer.step()` de fato altera); branco = valor
intermediário calculado a partir de outros (produto, soma, logit); verde =
`loss` e a probabilidade da classe alvo (`prob[j]` destacado), ou seja, o
ponto onde `.backward()` começa e o dado que ele está tentando reduzir.

**Como ler os números:** o gradiente que chega em cada `logit[j]` se
distribui de dois jeitos diferentes ao passar pra trás:
- Pela **soma** (`termo0 + termo1 + termo2 + termo3 + bias`): todo mundo
  recebe o **mesmo** gradiente (regra da soma, derivada = 1 pra cada parcela).
- Pela **multiplicação** (`e_i * w[j][i]`): cada peso recebe o gradiente do
  termo multiplicado pelo *outro* fator (regra do produto) — por isso
  `w[0][0].grad` (`0.0641`) é diferente de `w[0][1].grad` (`0.6622`), mesmo
  os dois termos tendo recebido o mesmo gradiente da soma.

## Parte 2 — intervenção

`intervention_demo()` roda o mesmo exemplo duas vezes com a mesma
inicialização de pesos (`torch.manual_seed(0)`), mudando só o `target`: uma
vez classe 1, outra vez classe 0.

```
                          baseline   intervenção
embedding.grad[0]            0.0579       -0.0503
embedding.grad[1]            0.0440       -0.0383
embedding.grad[2]           -0.0561        0.0487
embedding.grad[3]           -0.0195        0.0169
w[0][0].grad             0.0641       -0.0557
w[0][1].grad             0.6622       -0.5755
...
```

Como o forward é idêntico (mesmos pesos, mesma palavra), `logits` e `probs`
não mudam — só a `loss` e todos os gradientes trocam de sinal (não
exatamente, já que as probabilidades não são simétricas). Isso é esperado:
para uma perda de cross-entropy com softmax, `∂loss/∂logit_j = prob[j] - 1`
se `j` for a classe alvo, ou só `prob[j]` caso contrário — mudar qual classe
é a "certa" muda qual logit recebe o `-1`, invertendo o sinal do erro que
se propaga pra trás pra tudo que depende dele.

`gradient_step_intervencao.png` mostra esse segundo cenário no mesmo
formato visual da parte 1 (mesma legenda de cores acima), com a nova
classe alvo destacada em verde.

## Parte 3 — exemplo prático: embeddings pré-treinados numa tarefa real

`word2vec_transfer.py` treina um CBOW (`cbow.py`, mas apontado pro corpus de
sentimento em vez do `TOY_CORPUS` de gato/cachorro — senão os vocabulários
não bateriam) e usa a tabela de embeddings resultante pra **inicializar**
o classificador de sentimento do `simple_embedding.py`, em vez de pesos
aleatórios. É a mesma ideia por trás de usar word2vec/GloVe prontos numa
tarefa de classificação de texto: o modelo não parte do zero.

Comparando a perda do classificador com inicialização aleatória vs.
pré-treinada (mesma seed, mesmo tudo, só o embedding inicial muda):

```
 epoch     aleatório    pré-treinado
     1        0.8510          0.8103
     5        0.7574          0.7033
    10        0.6836          0.6397
    25        0.5837          0.5454
    50        0.5048          0.4442
   100        0.3835          0.3086
   200        0.1971          0.1585
```

A versão pré-treinada tem perda menor em **todas** as épocas — o embedding
pré-treinado já carrega alguma noção de como as palavras se relacionam
(mesmo aprendida num corpus pequeno), então o classificador converge mais
rápido do que partindo de vetores aleatórios.
