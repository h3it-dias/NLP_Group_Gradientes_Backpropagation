# pytorch_redes

Os mesmos conceitos de grafo computacional e `.backward()` do `micrograd/`,
só que com tensores e autograd **reais** do PyTorch, em vez de um motor
escrito à mão.

## Estrutura

- `derivacao_automatica.py` — como o autograd do PyTorch constrói o grafo
  computacional (`build_loss_graph`), o que `.grad_fn` guarda em cada
  tensor, o que `.backward()` de fato calcula (`compute_gradients`), como
  `torch.no_grad()` desliga o rastreamento de gradiente, e o produto
  Jacobiano-vetor para saídas não-escalares (`jacobian_product_demo`).
- `neural_network.py` — uma rede feed-forward simples
  (`Flatten → Linear → ReLU → Linear → ReLU → Linear`) para classificação
  de imagens, com uma exploração passo a passo do que cada camada
  (`nn.Flatten`, `nn.Linear`, `nn.ReLU`, `nn.Sequential`) faz com um tensor.

## Como executar

Com o `.venv` ativado, a partir da raiz do repositório:

```bash
python3 -m pytorch_redes.derivacao_automatica
python3 -m pytorch_redes.neural_network
```

Nenhuma dependência além de `torch` (já declarada em `pyproject.toml`).
