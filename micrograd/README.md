# micrograd

Motor de autograd escalar construído do zero (baseado no [micrograd](https://github.com/karpathy/micrograd) do Andrej Karpathy), com uma bibliotequinha de rede neural em cima dele. É a versão "artesanal" do que `pytorch_redes/derivacao_automatica.py` faz usando o autograd real do PyTorch.

## Estrutura

- `engine.py` — a classe `Value`: cada operação escalar (`+`, `*`, `**`, `tanh`, `exp`, ...) monta um nó no grafo computacional e sabe propagar seu próprio gradiente local. `.backward()` percorre o grafo em ordem topológica reversa aplicando a regra da cadeia.
- `nn.py` — `Neuron`, `Layer`, `MLP`, construídos em cima de `Value` (equivalente artesanal a `nn.Linear`/`nn.Sequential`).
- `visualize.py` — `trace`/`draw_dot`: desenha o grafo computacional com [graphviz](https://graphviz.org/).
- `derivative_demo.py` — motivação da derivada: estima `f'(x)` numericamente pela definição de limite, antes de qualquer `Value` entrar em cena.
- `demo.py` — neurônio manual, o mesmo neurônio com `tanh` expandido via `exp()`, checagem cruzada com o autograd real do PyTorch, e treino de um `MLP` por gradiente descendente manual.

## Como executar

Com o `.venv` do projeto ativado (`source .venv/bin/activate`), a partir da raiz do repositório:

```bash
python3 -m micrograd.derivative_demo
python3 -m micrograd.demo
```

Dependências: `numpy`, `matplotlib`, `graphviz` (pacote Python **e** o binário `dot` do sistema — `sudo apt install graphviz`) e `torch` (só usado na checagem cruzada do `demo.py`).

Os diagramas gerados vão para `micrograd/diagrams/` (pasta ignorada pelo git — são artefatos, não código).

## O que sai no terminal

### `derivative_demo.py`

```
f'(x) em x=0.6667 pela definição de limite: 0.0
d1 = 4.0
d2 = 3.999699999999999
slope (∂/∂a) = -3.000000000010772
Gráfico de f(x) salvo em .../micrograd/diagrams/f_x.png
```

- A primeira linha estima `f'(x)` em `x = 2/3` para `f(x) = 3x² - 4x + 5` usando `(f(x+h) - f(x))/h`. O resultado é `0.0` porque `x = 2/3` é exatamente o vértice da parábola (mínimo) — a derivada analítica ali é zero de verdade, não é erro numérico.
- As linhas seguintes fazem o mesmo para uma expressão com 3 variáveis (`a*b + c`), perturbando só `a`: o `slope` calculado (`≈ -3.0`) é a derivada parcial `∂/∂a`, que bate com o valor exato (`b = -3.0`).

### `demo.py`

O script imprime 4 seções, separadas por `----`:

**1. "Neurônio manual (tanh)"** — monta um neurônio (`x1,x2,w1,w2,b`), aplica `tanh`, chama `.backward()` e imprime `o = 0.7071` (a saída do neurônio) e o caminho do `.png` salvo com o grafo.

**2. "Neurônio com tanh expandido via exp"** — o mesmo neurônio, mas em vez de chamar `.tanh()` direto, calcula a mesma função decompondo em `exp()`, subtração e divisão (`tanh(x) = (e^2x - 1)/(e^2x + 1)`). Sai o mesmo `o = 0.7071` — confirma que decompor a operação não muda o resultado nem o gradiente.

**3. "Checagem cruzada com PyTorch"** — refaz o mesmo neurônio com `torch.Tensor(requires_grad=True)` e compara:
```
o = 0.7071067811865476
x2.grad = 0.49999999999999994
w2.grad = 0.0
x1.grad = -1.4999999999999998
w1.grad = 0.9999999999999999
```
Esses valores devem ser essencialmente os mesmos `grad` que aparecem no diagrama do neurônio manual (seção 1) — é a prova de que o motor artesanal (`Value`) calcula os gradientes corretamente, batendo com o autograd de verdade do PyTorch.

**4. "Treinando um MLP com gradiente descendente manual"** — treina um `MLP(3, [4, 4, 1])` por 40 épocas num dataset de 4 exemplos, imprimindo a perda (MSE) a cada uma:
```
epoch  0 | loss 5.784896
epoch  1 | loss 3.045487
...
epoch 39 | loss 0.031077
```
Os pesos são inicializados aleatoriamente (sem seed fixa), então os números exatos variam a cada execução — o que importa é a tendência: a perda deve cair de forma consistente até um valor bem baixo (a rede aprendeu a separar os 4 exemplos).

## Como interpretar as imagens

### `single_neuron.png` / `expanded_tanh_neuron.png`

Cada nó retangular representa um `Value` do grafo, no formato `{ rótulo | data valor | grad valor }`:

- **`data`** — o valor numérico calculado nesse ponto do forward pass.
- **`grad`** — `∂o/∂(esse valor)`, ou seja, o quanto a saída final `o` mudaria se esse valor mudasse em 1 unidade. É preenchido por `.backward()`.

Os nós em formato de elipse (sem retângulo) são as **operações** (`*`, `+`, `tanh`) que conectam os `Value`s — eles não guardam `data`/`grad` próprios, só indicam qual operação gerou o `Value` à direita deles.

O layout é **esquerda → direita** (`rankdir=LR`), seguindo a ordem do forward pass: das entradas/pesos (esquerda) até a saída `o` (direita, com `grad = 1.0` — é a raiz do `.backward()`, então `∂o/∂o = 1`).

Pra ler o efeito de cada parâmetro no resultado, olhe o `grad` dele: por exemplo, `w2` aparece com `grad 0.0000` porque `x2 = 0.0000` — como a contribuição de `w2` passa por `x2*w2`, e `x2` é zero, mudar `w2` não muda nada na saída (regra do produto: `d(x2*w2)/dw2 = x2`).

> Detalhe: o rótulo `"x1*w2 + x2*w2"` no meio do diagrama tem um erro de digitação herdado do notebook original do Karpathy (devia ser `"x1*w1 + x2*w2"`) — é só o texto do rótulo, não afeta o cálculo.

### `f_x.png`

Gráfico de `f(x) = 3x² - 4x + 5` para `x` entre -5 e 5 — uma parábola com mínimo em `x = 2/3` (onde `derivative_demo.py` mostra que a derivada numérica dá `0.0`). Serve só pra visualizar por que faz sentido a derivada zerar exatamente ali: é o ponto mais baixo da curva.
