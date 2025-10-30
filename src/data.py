import os
import pickle
from typing import Tuple, List
from torch.utils.data import Dataset, Subset
import torchvision
from torchvision import transforms as T
import torch

# Função auxiliar para carregar arquivos pkl do CIFAR-100
def _pickle_load(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f, encoding="latin1")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Arquivo CIFAR-100 não encontrado em: {path}") from e
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar pickle em {path}: {e}") from e


class CIFAR100MultiTask(Dataset):
    def __init__(self, root: str, train: bool = True, transform=None, download: bool = True):
        # Dataset oficial do torchvision
        self.base_dataset = torchvision.datasets.CIFAR100(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )
        self.transform = transform
        self.train = train

        # Aqui nesse trecho verifico se os nomes das classes estão disponíveis

        # Local dos arquivos .pkl do CIFAR-100
        self.base_folder = getattr(self.base_dataset, "base_folder", "cifar-100-python")
        self.dataset_dir = os.path.join(root, self.base_folder)

        # Carrega nomes (fine: classes e coarse: superclasses) do meta
        meta_path = os.path.join(self.dataset_dir, "meta")
        meta = _pickle_load(meta_path)
        self.fine_label_names: List[str] = list(meta["fine_label_names"])
        self.coarse_label_names: List[str] = list(meta["coarse_label_names"])

        # Carrega coarse_labels do split correspondente (train/test)
        split_file = "train" if train else "test"
        split_path = os.path.join(self.dataset_dir, split_file)
        split_data = _pickle_load(split_path)
        self._coarse_targets: List[int] = list(split_data["coarse_labels"])

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        # img e fine label do torchvision
        img, fine_label = self.base_dataset[idx]
        # coarse label lido direto do arquivo do CIFAR-100
        coarse_label = int(self._coarse_targets[idx])
        return img, (int(fine_label), coarse_label)

    def get_label_names(self) -> Tuple[List[str], List[str]]:
        return self.fine_label_names, self.coarse_label_names


# Transforms padrão para CIFAR-100
def _default_transforms():
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    return T.Compose([T.ToTensor(), T.Normalize(mean, std)])


def get_train_val_test_datasets(
    root: str,
    val_size: int = 5000,
    seed: int = 42,
    download: bool = True,
    transform_train=None,
    transform_val=None,
    transform_test=None,
):
    """
    Retorna (train_ds, val_ds, test_ds).
    - val_size: número de amostras para validação retiradas do split 'train' do CIFAR-100.
    - As transforms podem ser diferentes para treino/val/test.
    """
    if transform_train is None:
        transform_train = _default_transforms()
    if transform_val is None:
        transform_val = _default_transforms()
    if transform_test is None:
        transform_test = _default_transforms()

    # Cria uma instância para obter o tamanho total do split de treino original
    train_full = CIFAR100MultiTask(root=root, train=True, transform=None, download=download)
    n = len(train_full)
    if val_size < 0 or val_size >= n:
        raise ValueError(f"val_size deve estar em [0, {n-1}]")

    # Gera índices de forma reprodutível
    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    train_idx = perm[: n - val_size]
    val_idx = perm[n - val_size :]

    # Cria duas instâncias separadas para permitir transforms diferentes
    train_ds = Subset(CIFAR100MultiTask(root=root, train=True, transform=transform_train, download=False), train_idx)
    val_ds = Subset(CIFAR100MultiTask(root=root, train=True, transform=transform_val, download=False), val_idx)

    # Test dataset permanece com train=False
    test_ds = CIFAR100MultiTask(root=root, train=False, transform=transform_test, download=False)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=None)  # Padrão: <repo>/data
    parser.add_argument("--show-sample", action="store_true")
    parser.add_argument("--val-size", type=int, default=5000, help="Número de amostras para validação (padrão 5000)")
    parser.add_argument("--seed", type=int, default=42, help="Semente para split reprodutível")
    args = parser.parse_args()

    # Define padrão <repo>/data se não for passado
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    root = args.root or os.path.join(repo_root, "data")

    print(f"Usando root: {root}")
    train_ds, val_ds, test_ds = get_train_val_test_datasets(
        root=root,
        val_size=args.val_size,
        seed=args.seed,
        download=True,
    )

    print(f"Tamanho treino: {len(train_ds)}")
    print(f"Tamanho validação: {len(val_ds)}")
    print(f"Tamanho teste:  {len(test_ds)}")

    # Obtém nomes de classes a partir da instância do dataset 'base' usada nos Subset
    # Se train_ds for Subset, pega train_ds.dataset; se for CIFAR100MultiTask direto, usa ele.
    base_for_names = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
    fine_names, coarse_names = base_for_names.get_label_names()

    print(f"\nSuperclasses ({len(coarse_names)}):")
    for i, name in enumerate(coarse_names):
        print(f"  {i:2d} - {name}")

    print(f"\nClasses finas ({len(fine_names)}):")
    for i, name in enumerate(fine_names):
        print(f"  {i:2d} - {name}")

    if args.show_sample:
        # Mostra uma amostra do dataset de treino (usa índices do Subset para acessar)
        sample_idx = 0
        ds_for_sample = train_ds
        if isinstance(train_ds, Subset):
            ds_for_sample = train_ds.dataset
            real_idx = train_ds.indices[sample_idx]
            x, (y_fine, y_coarse) = ds_for_sample[real_idx]
        else:
            x, (y_fine, y_coarse) = train_ds[sample_idx]

        print("\nAmostra:")
        if hasattr(x, "shape"):
            print(f"  x shape: {tuple(x.shape)}")
        else:
            print(f"  x type: {type(x)}")
        print(f"  y_fine: {y_fine}, y_coarse: {y_coarse}")