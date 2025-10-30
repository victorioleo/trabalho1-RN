# trabalho1-RN
Trabalho 1 da disciplina Redes Neurais - FACOM/UFMS

## Descrição do Projeto

Este repositório implementa três modelos de classificação para o dataset CIFAR-100:

1. **Modelo de 20 Superclasses**: Classifica imagens em 20 categorias gerais (superclasses)
2. **Modelo de 100 Classes**: Classifica imagens em 100 categorias específicas (classes fine)
3. **Modelo Multi-head**: Possui duas cabeças de classificação simultâneas:
   - Uma cabeça para 20 superclasses
   - Uma cabeça para 100 classes
   - A loss final é a soma das losses das duas cabeças: `loss_total = loss_coarse + loss_fine`

Todos os modelos utilizam arquiteturas do `timm` ou `torchvision.models` limitadas a 10 milhões de parâmetros.

## Estrutura do Projeto

```
.
├── cifar100_labels.py    # Mapeamento de classes fine para superclasses
├── dataset.py            # Dataset e dataloaders
├── models.py            # Definições dos três modelos
├── train.py             # Script de treinamento
├── config.py            # Configurações de exemplo
└── requirements.txt     # Dependências do projeto
```

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/victorioleo/trabalho1-RN.git
cd trabalho1-RN
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Como Usar

### Treinar Modelo de 20 Superclasses

```bash
python train.py --model_type coarse --epochs 100 --batch_size 128
```

### Treinar Modelo de 100 Classes

```bash
python train.py --model_type fine --epochs 100 --batch_size 128
```

### Treinar Modelo Multi-head

```bash
python train.py --model_type multihead --epochs 100 --batch_size 128
```

### Argumentos Disponíveis

- `--model_type`: Tipo de modelo (`coarse`, `fine`, ou `multihead`)
- `--backbone`: Arquitetura base do timm (default: `mobilenetv3_small_100`)
- `--pretrained`: Usar pesos pré-treinados
- `--epochs`: Número de épocas (default: 100)
- `--batch_size`: Tamanho do batch (default: 128)
- `--lr`: Learning rate inicial (default: 0.1)
- `--momentum`: Momentum para SGD (default: 0.9)
- `--weight_decay`: Weight decay (default: 5e-4)
- `--num_workers`: Workers para carregamento de dados (default: 4)
- `--data_dir`: Diretório para armazenar o dataset (default: ./data)

### Testar os Modelos

Para verificar a criação dos modelos e suas dimensões:

```bash
python models.py
```

## Dataset CIFAR-100

O CIFAR-100 possui:
- 50,000 imagens de treino
- 10,000 imagens de teste
- 100 classes fine agrupadas em 20 superclasses
- Imagens de 32x32 pixels RGB

### Superclasses

As 20 superclasses são:
1. aquatic mammals
2. fish
3. flowers
4. food containers
5. fruit and vegetables
6. household electrical devices
7. household furniture
8. insects
9. large carnivores
10. large man-made outdoor things
11. large natural outdoor scenes
12. large omnivores and herbivores
13. medium-sized mammals
14. non-insect invertebrates
15. people
16. reptiles
17. small mammals
18. trees
19. vehicles 1
20. vehicles 2

## Arquiteturas Suportadas

O projeto usa `mobilenetv3_small_100` por padrão (~2.5M parâmetros), mas suporta outras arquiteturas do timm:

- `mobilenetv3_small_100` - ~2.5M params
- `mobilenetv3_large_100` - ~5.5M params
- `efficientnet_b0` - ~5.3M params

Para usar uma arquitetura diferente:
```bash
python train.py --model_type fine --backbone efficientnet_b0
```

## Checkpoints

Os melhores modelos são salvos automaticamente em `checkpoints/` durante o treinamento.

## Licença

Este projeto foi desenvolvido para fins educacionais como parte da disciplina Redes Neurais - FACOM/UFMS.
