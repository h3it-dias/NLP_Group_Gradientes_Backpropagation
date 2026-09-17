# notebooks

Versões Jupyter/Colab dos conceitos do repositório. Todos são
autocontidos — nenhum importa de `micrograd/`, `pytorch_redes/` ou
`embeddings/` — então abrem e rodam direto no Colab, sem precisar clonar
o repositório.

## `backprop_do_zero.ipynb` — referência do projeto

Motor de autograd escalar construído do zero (`Value`, `draw_dot`,
`Neuron`/`Layer`/`MLP`, treino), culminando na entrega prática: um passo
de aprendizado numa rede mínima (embedding + linear) com duas
intervenções — trocar o alvo da predição, trocar um peso à mão — mostrando
o efeito de cada uma sobre o gradiente.

## Tutoriais de fundamentos do PyTorch

- **`Quick_Start_Pytorch.ipynb`** — ciclo de treino completo (dados,
  `DataLoader`, modelo, otimizador, treino/teste, salvar/carregar) numa
  rede feed-forward classificando o FashionMNIST. É o único notebook do
  repositório com treino de ponta a ponta em dados reais — os demais usam
  corpora de brinquedo.
- **`Introducao_torch_autograd.ipynb`** — forward → backward →
  `optimizer.step()` num modelo pré-treinado (ResNet-18); a versão mais
  curta de "como o autograd do PyTorch funciona na prática".
- **`Tensores_pytorch.ipynb`** — fundamentos de tensores (criação,
  atributos, indexação, operações aritméticas). Inacabado — para no meio
  do tutorial oficial de tensores do PyTorch que segue, antes de operações
  in-place e da ponte com NumPy.

## `estudos/`

Iterações anteriores da entrega "passo mínimo + intervenção", mantidas
como material de estudo/teste — ver [`estudos/README.md`](estudos/).
