# Trabalho 1 – Modelo Multi-head
Disciplina: Redes Neurais – FACOM/UFMS

Este trabalho consiste em treinar e comparar três modelos de classificação de imagens usando o conjunto de dados CIFAR-100, incluindo um modelo multi-head que aprende simultaneamente as superclasses e as classes.

## CIFAR-100
- 60.000 imagens (32x32 RGB): 50.000 treino, 10.000 teste.
- 100 classes finas organizadas em 20 superclasses.
- Download via `torchvision.datasets.CIFAR100`.

Superclasses e classes:
- aquatic mammals: beaver, dolphin, otter, seal, whale
- fish: aquarium fish, flatfish, ray, shark, trout
- flowers: orchids, poppies, roses, sunflowers, tulips
- food containers: bottles, bowls, cans, cups, plates
- fruit and vegetables: apples, mushrooms, oranges, pears, sweet peppers
- household electrical devices: clock, computer keyboard, lamp, telephone, television
- household furniture: bed, chair, couch, table, wardrobe
- insects: bee, beetle, butterfly, caterpillar, cockroach
- large carnivores: bear, leopard, lion, tiger, wolf
- large man-made outdoor things: bridge, castle, house, road, skyscraper
- large natural outdoor scenes: cloud, forest, mountain, plain, sea
- large omnivores and herbivores: camel, cattle, chimpanzee, elephant, kangaroo
- medium-sized mammals: fox, porcupine, possum, raccoon, skunk
- non-insect invertebrates: crab, lobster, snail, spider, worm
- people: baby, boy, girl, man, woman
- reptiles: crocodile, dinosaur, lizard, snake, turtle
- small mammals: hamster, mouse, rabbit, shrew, squirrel
- trees: maple, oak, palm, pine, willow
- vehicles 1: bicycle, bus, motorcycle, pickup truck, train
- vehicles 2: lawn-mower, rocket, streetcar, tank, tractor

Normalização recomendada (CIFAR-100):
- mean = [0.5071, 0.4867, 0.4408]
- std = [0.2675, 0.2565, 0.2761]

Augmentations típicas:
- RandomCrop(32, padding=4), RandomHorizontalFlip(), ToTensor(), Normalize(mean, std).

## Modelos a treinar
1) Classificador de 20 superclasses  
2) Classificador de 100 classes  
3) Modelo multi-head:
   - Backbone comum + flatten.
   - Duas cabeças (MLPs separadas):
     - Head A: 20 superclasses.
     - Head B: 100 classes.
   - Cada head com sua CrossEntropyLoss.
   - Loss final: `loss_total = loss_super + loss_fina` (soma simples).

## Restrições de arquitetura
- Usar modelos do `timm` ou `torchvision.models`.
- Limite de até 10 milhões de parâmetros.
- Exemplos que se encaixam no limite:
  - torchvision: `mobilenet_v3_small`, `shufflenet_v2_x1_0`, `squeezenet1_0`, `efficientnet_b0`
  - timm: variantes leves (ex.: `efficientnet_b0`, `mobilenet_v3_small`).
- Para contar parâmetros, da para usar (PyTorch): `sum(p.numel() for p in model.parameters() if p.requires_grad)`.

## Divisão do dataset
- Treino, Validação, Teste.
- Sugestão:
  - Treino: 45.000 (a partir de `train=True`)
  - Validação: 5.000 (a partir de `train=True`)
  - Teste: 10.000 (usar `test=True`)

## Requisitos do treinamento
- Exibir curvas de loss por época para treino e validação.
- Checkpoint: salvar o melhor modelo com base na loss de validação (a cada melhoria).
- Early stopping com paciência configurável.
- Após o treinamento, apresentar `classification_report` do scikit-learn:
  - Para 20 superclasses (quando aplicável).
  - Para 100 classes.

## Saídas esperadas
- Gráficos de loss (treino/val).
- Pesos salvos do melhor modelo por experimento.
- Classification report no console/arquivo para:
  - Modelo de 20 superclasses.
  - Modelo de 100 classes.
  - Modelo multi-head (para cada head).

## Organização sugerida do projeto
- `src/data.py` — dataset, transforms e splits.
- `src/models.py` — criação do backbone e heads (20/100/multi-head).
- `src/train.py` — loop de treino/val, early stopping, checkpoint.
- `src/eval.py` — avaliação e classification report.
- `configs/` — hiperparâmetros por experimento (opcional).
- `runs/` — logs (ex.: TensorBoard) e checkpoints.

## Como executar (exemplo em Windows)
Instalação:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision timm scikit-learn matplotlib tensorboard
```

Treino (exemplos hipotéticos):
```powershell
# 20 superclasses
python src\train.py --dataset cifar100 --task super --model mobilenet_v3_small --epochs 100 --batch-size 128 --patience 10 --lr 0.001

# 100 classes
python src\train.py --dataset cifar100 --task fine --model shufflenet_v2_x1_0 --epochs 100 --batch-size 128 --patience 10 --lr 0.001

# multi-head (20 + 100)
python src\train.py --dataset cifar100 --task multi --model efficientnet_b0 --epochs 120 --batch-size 128 --patience 12 --lr 0.001
```

Visualizar curvas com TensorBoard:
```powershell
tensorboard --logdir runs
```

---
