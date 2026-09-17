# Estudos

Notebooks de iteração — mantidos como material de estudo/teste, mas
**superados** pela versão de apresentação em `notebooks/backprop_do_zero.ipynb`
para o mesmo conteúdo:

- **`micrograd.ipynb`** — motor de autograd escalar (`Value`, `draw_dot`,
  neurônio, MLP, treino). O mesmo conteúdo está nas seções 1–7 de
  `backprop_do_zero.ipynb`, e como pacote importável em `micrograd/`.
- **`Embeddings_Gradientes.ipynb`** e **`Embeddings_Gradientes_Simples.ipynb`**
  — duas iterações de "passo mínimo de aprendizado + intervenção" usando
  `nn.Linear`/`nn.Embedding` do PyTorch, decompostos escalar por escalar
  pra dar pra desenhar cada peso. `backprop_do_zero.ipynb` faz a mesma
  entrega (e com duas intervenções, não uma) usando o motor `Value` em vez
  de PyTorch — sem precisar decompor nada, porque cada `Value` já é
  escalar por natureza.

Nenhum desses arquivos está quebrado; ficaram aqui só pra não competir com
a versão de apresentação como "a" referência do projeto.
