import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from sklearn.metrics import classification_report
from tqdm import tqdm

from data import get_train_val_test_datasets
from models import create_model


def get_transforms():
    """Transforms para teste (sem augmentation)."""
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def evaluate(args):
    """Avalia modelo salvo no conjunto de teste."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")
    
    # Dataset de teste
    print("Carregando dataset de teste...")
    transform_test = get_transforms()
    _, _, test_ds = get_train_val_test_datasets(
        root=args.data_root,
        val_size=5000,
        seed=42,
        download=False,
        transform_test=transform_test,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Nomes das classes
    from torch.utils.data import Subset
    base_ds = test_ds.dataset if isinstance(test_ds, Subset) else test_ds
    fine_names, coarse_names = base_ds.get_label_names()
    
    # Modelo
    print(f"Carregando modelo para task: {args.task}")
    model = create_model(task=args.task, backbone=args.backbone, pretrained=False)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Checkpoint carregado: época {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")
    
    # Inferência
    all_preds_fine = []
    all_labels_fine = []
    all_preds_coarse = []
    all_labels_coarse = []
    
    print("Executando inferência...")
    with torch.no_grad():
        for images, (labels_fine, labels_coarse) in tqdm(test_loader):
            images = images.to(device)
            
            if args.task == "superclass":
                outputs = model(images)
                _, preds = outputs.max(1)
                all_preds_coarse.extend(preds.cpu().numpy())
                all_labels_coarse.extend(labels_coarse.numpy())
                
            elif args.task == "fine":
                outputs = model(images)
                _, preds = outputs.max(1)
                all_preds_fine.extend(preds.cpu().numpy())
                all_labels_fine.extend(labels_fine.numpy())
                
            elif args.task == "multihead":
                out_coarse, out_fine = model(images)
                _, preds_coarse = out_coarse.max(1)
                _, preds_fine = out_fine.max(1)
                
                all_preds_coarse.extend(preds_coarse.cpu().numpy())
                all_labels_coarse.extend(labels_coarse.numpy())
                all_preds_fine.extend(preds_fine.cpu().numpy())
                all_labels_fine.extend(labels_fine.numpy())
    
    # Classification reports
    print("\n" + "="*80)
    if args.task == "superclass" or args.task == "multihead":
        print("CLASSIFICATION REPORT - SUPERCLASSES (20 classes)")
        print("="*80)
        print(classification_report(
            all_labels_coarse,
            all_preds_coarse,
            target_names=coarse_names,
            digits=4,
        ))
    
    if args.task == "fine" or args.task == "multihead":
        print("\n" + "="*80)
        print("CLASSIFICATION REPORT - CLASSES FINAS (100 classes)")
        print("="*80)
        print(classification_report(
            all_labels_fine,
            all_preds_fine,
            target_names=fine_names,
            digits=4,
        ))
    
    # Salva relatórios em arquivo
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write(f"AVALIAÇÃO - TASK: {args.task}\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write("="*80 + "\n\n")
            
            if args.task == "superclass" or args.task == "multihead":
                f.write("CLASSIFICATION REPORT - SUPERCLASSES (20 classes)\n")
                f.write("="*80 + "\n")
                f.write(classification_report(
                    all_labels_coarse,
                    all_preds_coarse,
                    target_names=coarse_names,
                    digits=4,
                ))
                f.write("\n")
            
            if args.task == "fine" or args.task == "multihead":
                f.write("\n" + "="*80 + "\n")
                f.write("CLASSIFICATION REPORT - CLASSES FINAS (100 classes)\n")
                f.write("="*80 + "\n")
                f.write(classification_report(
                    all_labels_fine,
                    all_preds_fine,
                    target_names=fine_names,
                    digits=4,
                ))
        
        print(f"\nRelatório salvo em: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data", help="Diretório dos dados")
    parser.add_argument("--task", type=str, required=True, choices=["superclass", "fine", "multihead"])
    parser.add_argument("--backbone", type=str, default="mobilenetv3_small_100")
    parser.add_argument("--checkpoint", type=str, required=True, help="Caminho do checkpoint (.pth)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-file", type=str, default=None, help="Salvar relatório em arquivo .txt")
    
    args = parser.parse_args()
    evaluate(args)