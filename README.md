# Gradientes e Backpropagation

Repositório de estudo sobre como redes neurais aprendem: derivação
automática (autograd), a regra da cadeia, e o que `.backward()` e
`optimizer.step()` realmente fazem por baixo dos panos. O conteúdo avança
em camadas de abstração — de um motor de autograd escalar escrito à mão até
embeddings de palavras (Word2Vec) treinados com PyTorch.

## Destaque: `notebooks/backprop_do_zero.ipynb`

Este é o notebook de referência do projeto. Constrói um motor de autograd
escalar do zero (`Value`, inspirado no
[micrograd](https://github.com/karpathy/micrograd) do Andrej Karpathy),
desenha o grafo computacional peso por peso, e termina numa entrega
prática: um passo de aprendizado numa rede mínima (embedding + linear),
com duas intervenções — mudar o alvo da predição e mudar um peso à mão —
mostrando o efeito de cada uma sobre o gradiente.

**[Abrir no Colab](https://colab.research.google.com/github/h3it-dias/NLP_Group_Gradientes_Backpropagation/blob/main/notebooks/backprop_do_zero.ipynb)**

## Estrutura

| Pasta | Conteúdo |
|---|---|
| [`micrograd/`](micrograd/) | Motor de autograd escalar construído do zero: `Value`, `Neuron`/`Layer`/`MLP`, visualização do grafo com graphviz. |
| [`pytorch_redes/`](pytorch_redes/) | Os mesmos conceitos com autograd real do PyTorch (tensores, `.grad_fn`, `.backward()`), mais uma rede feed-forward de exemplo. |
| [`embeddings/`](embeddings/) | Três arquiteturas de embedding de palavras (classificador simples, Word2Vec CBOW e Skip-gram) e um exemplo de embeddings pré-treinados alimentando uma tarefa real. |
| [`notebooks/`](notebooks/) | Versões Jupyter/Colab desses conceitos. `backprop_do_zero.ipynb` é a referência atual; `estudos/` guarda iterações anteriores, mantidas como material de estudo. |

Cada pasta acima tem seu próprio `README.md` com mais detalhes.

## Como rodar

### 1. Ambiente virtual e dependências

Recomendado: [`uv`](https://docs.astral.sh/uv/) — gerencia o ambiente
virtual e um lockfile automaticamente, e instala bem mais rápido que pip:

```bash
uv sync
source .venv/bin/activate
```

Alternativa com pip puro:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Dependência de sistema: Graphviz

Os módulos de visualização desenham grafos computacionais com
[graphviz](https://graphviz.org/), que precisa do binário `dot` instalado
no sistema (o pacote Python sozinho não é suficiente):

```bash
sudo apt install graphviz    # Debian/Ubuntu
```

### 3. Rodar um módulo

Cada módulo é executável e imprime/desenha um resultado explicativo:

```bash
python3 -m micrograd.demo
python3 -m pytorch_redes.derivacao_automatica
python3 -m embeddings.cbow
```

Os diagramas gerados vão para `<pacote>/diagrams/` (artefatos, não
versionados).

### 4. Abrir um notebook

```bash
jupyter notebook notebooks/backprop_do_zero.ipynb
```

Todos os notebooks são autocontidos — nenhum depende de clonar o
repositório ou importar os pacotes acima, então também abrem direto pelo
link do Colab.

## Desenvolvimento

Lint com [ruff](https://docs.astral.sh/ruff/):

```bash
uv sync --group dev
uv run ruff check .
```

## Licença

Ainda não definida.
