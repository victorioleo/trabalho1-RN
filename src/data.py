import os
import pickle
from typing import Tuple, List
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms as T

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


if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=None) # Padrão: <repo>/data, mas pode ser definida por argumento
    parser.add_argument("--show-sample", action="store_true") # Mostra uma amostra do dataset, tamanho do x e rótulos
    args = parser.parse_args()

    # Define padrão <repo>/data se não for passado
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    root = args.root or os.path.join(repo_root, "data") # muda para o argumento se fornecido

    transform = _default_transforms()

    print(f"Usando root: {root}")
    train_ds = CIFAR100MultiTask(root=root, train=True, transform=transform, download=True)
    test_ds = CIFAR100MultiTask(root=root, train=False, transform=transform, download=True)

    print(f"Tamanho treino: {len(train_ds)}")
    print(f"Tamanho teste:  {len(test_ds)}")

    fine_names, coarse_names = train_ds.get_label_names()

    print(f"\nSuperclasses ({len(coarse_names)}):")
    for i, name in enumerate(coarse_names):
        print(f"  {i:2d} - {name}")

    print(f"\nClasses finas ({len(fine_names)}):")
    for i, name in enumerate(fine_names):
        print(f"  {i:2d} - {name}")

    if args.show_sample:
        x, (y_fine, y_coarse) = train_ds[0]
        print("\nAmostra:")
        if isinstance(x, torch.Tensor):
            print(f"  x shape: {tuple(x.shape)}")
        else:
            print(f"  x type: {type(x)}")
        print(f"  y_fine: {y_fine}, y_coarse: {y_coarse}")