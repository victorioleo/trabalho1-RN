import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm

from data import get_train_val_test_datasets
from models import create_model, count_parameters


def get_transforms(augment=True):
    """Retorna transforms para treino e validação/teste."""
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    
    if augment:
        transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return transform


def train_epoch(model, loader, criterion, optimizer, device, task):
    """Treina uma época."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Train", leave=False)
    for images, (labels_fine, labels_coarse) in pbar:
        images = images.to(device)
        
        optimizer.zero_grad()
        
        if task == "superclass":
            labels = labels_coarse.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = outputs.max(1)
            
        elif task == "fine":
            labels = labels_fine.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = outputs.max(1)
            
        elif task == "multihead":
            labels_coarse = labels_coarse.to(device)
            labels_fine = labels_fine.to(device)
            out_coarse, out_fine = model(images)
            
            loss_coarse = criterion(out_coarse, labels_coarse)
            loss_fine = criterion(out_fine, labels_fine)
            loss = loss_coarse + loss_fine
            
            _, preds_coarse = out_coarse.max(1)
            _, preds_fine = out_fine.max(1)
            # Acurácia baseada na predição fine
            preds = preds_fine
            labels = labels_fine
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)
        
        pbar.set_postfix({"loss": loss.item(), "acc": 100.0 * correct / total})
    
    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


def validate_epoch(model, loader, criterion, device, task):
    """Valida uma época."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Val", leave=False)
        for images, (labels_fine, labels_coarse) in pbar:
            images = images.to(device)
            
            if task == "superclass":
                labels = labels_coarse.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, preds = outputs.max(1)
                
            elif task == "fine":
                labels = labels_fine.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, preds = outputs.max(1)
                
            elif task == "multihead":
                labels_coarse = labels_coarse.to(device)
                labels_fine = labels_fine.to(device)
                out_coarse, out_fine = model(images)
                
                loss_coarse = criterion(out_coarse, labels_coarse)
                loss_fine = criterion(out_fine, labels_fine)
                loss = loss_coarse + loss_fine
                
                _, preds = out_fine.max(1)
                labels = labels_fine
            
            total_loss += loss.item() * images.size(0)
            correct += preds.eq(labels).sum().item()
            total += images.size(0)
    
    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


def plot_curves(train_losses, val_losses, save_path):
    """Plota curvas de loss."""
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.title('Curvas de Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gráfico salvo em: {save_path}")


def train(args):
    """Função principal de treinamento."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")
    
    # Cria diretório de saída
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Datasets
    print("Carregando datasets...")
    transform_train = get_transforms(augment=True)
    transform_val = get_transforms(augment=False)
    
    train_ds, val_ds, test_ds = get_train_val_test_datasets(
        root=args.data_root,
        val_size=args.val_size,
        seed=args.seed,
        download=True,
        transform_train=transform_train,
        transform_val=transform_val,
        transform_test=transform_val,
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # Modelo
    print(f"Criando modelo para task: {args.task}")
    model = create_model(task=args.task, backbone=args.backbone, pretrained=args.pretrained)
    model = model.to(device)
    n_params = count_parameters(model)
    print(f"Parâmetros: {n_params:,}")
    if n_params > 10_000_000:
        print(f"[AVISO] Modelo excede 10M parâmetros!")
    
    # Loss e otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Treinamento
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    print(f"\nIniciando treinamento por {args.epochs} épocas (patience={args.patience})...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nÉpoca {epoch}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, args.task)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, args.task)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        
        # Checkpoint se melhorou
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = os.path.join(args.output_dir, f"best_model_{args.task}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"✓ Melhor modelo salvo (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{args.patience}")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping acionado na época {epoch}")
            break
    
    # Salva curvas
    plot_path = os.path.join(args.output_dir, f"loss_curves_{args.task}.png")
    plot_curves(train_losses, val_losses, plot_path)
    
    print(f"\nTreinamento concluído! Melhor val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Dados
    parser.add_argument("--data-root", type=str, default="data", help="Diretório dos dados")
    parser.add_argument("--val-size", type=int, default=5000, help="Tamanho do conjunto de validação")
    parser.add_argument("--seed", type=int, default=42, help="Semente para split")
    
    # Modelo
    parser.add_argument("--task", type=str, required=True, choices=["superclass", "fine", "multihead"], help="Tipo de modelo")
    parser.add_argument("--backbone", type=str, default="mobilenetv3_small_100", help="Backbone do timm")
    parser.add_argument("--pretrained", action="store_true", help="Usar pesos pré-treinados")
    
    # Treinamento
    parser.add_argument("--epochs", type=int, default=100, help="Número máximo de épocas")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=15, help="Patience para early stopping")
    parser.add_argument("--num-workers", type=int, default=4, help="Num workers para DataLoader")
    
    # Saída
    parser.add_argument("--output-dir", type=str, default="runs", help="Diretório de saída")
    
    args = parser.parse_args()
    train(args)