"""
Training script for CIFAR-100 models.

Supports three training modes:
1. coarse: Train on 20 superclasses
2. fine: Train on 100 classes
3. multihead: Train with both heads (loss = loss_coarse + loss_fine)
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import get_dataloaders
from models import create_model


def train_epoch_coarse(model, train_loader, criterion, optimizer, device):
    """Train one epoch for coarse model."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, _, coarse_labels in pbar:
        images = images.to(device)
        coarse_labels = coarse_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, coarse_labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += coarse_labels.size(0)
        correct += predicted.eq(coarse_labels).sum().item()
        
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
    
    return running_loss / len(train_loader), 100. * correct / total


def train_epoch_fine(model, train_loader, criterion, optimizer, device):
    """Train one epoch for fine model."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, fine_labels, _ in pbar:
        images = images.to(device)
        fine_labels = fine_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, fine_labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += fine_labels.size(0)
        correct += predicted.eq(fine_labels).sum().item()
        
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
    
    return running_loss / len(train_loader), 100. * correct / total


def train_epoch_multihead(model, train_loader, criterion, optimizer, device):
    """Train one epoch for multi-head model."""
    model.train()
    running_loss = 0.0
    running_coarse_loss = 0.0
    running_fine_loss = 0.0
    coarse_correct = 0
    fine_correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, fine_labels, coarse_labels in pbar:
        images = images.to(device)
        fine_labels = fine_labels.to(device)
        coarse_labels = coarse_labels.to(device)
        
        optimizer.zero_grad()
        coarse_outputs, fine_outputs = model(images)
        
        # Calculate losses for both heads
        coarse_loss = criterion(coarse_outputs, coarse_labels)
        fine_loss = criterion(fine_outputs, fine_labels)
        
        # Total loss is the sum of both losses
        loss = coarse_loss + fine_loss
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_coarse_loss += coarse_loss.item()
        running_fine_loss += fine_loss.item()
        
        _, coarse_predicted = coarse_outputs.max(1)
        _, fine_predicted = fine_outputs.max(1)
        total += fine_labels.size(0)
        coarse_correct += coarse_predicted.eq(coarse_labels).sum().item()
        fine_correct += fine_predicted.eq(fine_labels).sum().item()
        
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'coarse_acc': 100. * coarse_correct / total,
            'fine_acc': 100. * fine_correct / total
        })
    
    return (running_loss / len(train_loader), 
            100. * coarse_correct / total, 
            100. * fine_correct / total)


def evaluate_coarse(model, test_loader, criterion, device):
    """Evaluate coarse model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, _, coarse_labels in test_loader:
            images = images.to(device)
            coarse_labels = coarse_labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, coarse_labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += coarse_labels.size(0)
            correct += predicted.eq(coarse_labels).sum().item()
    
    return running_loss / len(test_loader), 100. * correct / total


def evaluate_fine(model, test_loader, criterion, device):
    """Evaluate fine model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, fine_labels, _ in test_loader:
            images = images.to(device)
            fine_labels = fine_labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, fine_labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += fine_labels.size(0)
            correct += predicted.eq(fine_labels).sum().item()
    
    return running_loss / len(test_loader), 100. * correct / total


def evaluate_multihead(model, test_loader, criterion, device):
    """Evaluate multi-head model."""
    model.eval()
    running_loss = 0.0
    coarse_correct = 0
    fine_correct = 0
    total = 0
    
    with torch.no_grad():
        for images, fine_labels, coarse_labels in test_loader:
            images = images.to(device)
            fine_labels = fine_labels.to(device)
            coarse_labels = coarse_labels.to(device)
            
            coarse_outputs, fine_outputs = model(images)
            
            coarse_loss = criterion(coarse_outputs, coarse_labels)
            fine_loss = criterion(fine_outputs, fine_labels)
            loss = coarse_loss + fine_loss
            
            running_loss += loss.item()
            _, coarse_predicted = coarse_outputs.max(1)
            _, fine_predicted = fine_outputs.max(1)
            total += fine_labels.size(0)
            coarse_correct += coarse_predicted.eq(coarse_labels).sum().item()
            fine_correct += fine_predicted.eq(fine_labels).sum().item()
    
    return (running_loss / len(test_loader), 
            100. * coarse_correct / total, 
            100. * fine_correct / total)


def train(args):
    """Main training function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Loading data...")
    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir
    )
    
    # Create model
    print(f"\nCreating {args.model_type} model...")
    model = create_model(
        model_type=args.model_type,
        model_name=args.backbone,
        pretrained=args.pretrained
    )
    model = model.to(device)
    
    # Create optimizer and criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )
    
    # Training loop
    best_acc = 0.0
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        if args.model_type == 'coarse':
            train_loss, train_acc = train_epoch_coarse(
                model, train_loader, criterion, optimizer, device
            )
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            
            # Evaluate
            test_loss, test_acc = evaluate_coarse(
                model, test_loader, criterion, device
            )
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                os.makedirs('checkpoints', exist_ok=True)
                torch.save(model.state_dict(), 
                          f'checkpoints/{args.model_type}_{args.backbone}_best.pth')
                print(f"Saved best model with accuracy: {best_acc:.2f}%")
                
        elif args.model_type == 'fine':
            train_loss, train_acc = train_epoch_fine(
                model, train_loader, criterion, optimizer, device
            )
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            
            # Evaluate
            test_loss, test_acc = evaluate_fine(
                model, test_loader, criterion, device
            )
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                os.makedirs('checkpoints', exist_ok=True)
                torch.save(model.state_dict(), 
                          f'checkpoints/{args.model_type}_{args.backbone}_best.pth')
                print(f"Saved best model with accuracy: {best_acc:.2f}%")
                
        else:  # multihead
            train_loss, train_coarse_acc, train_fine_acc = train_epoch_multihead(
                model, train_loader, criterion, optimizer, device
            )
            print(f"Train Loss: {train_loss:.4f}, "
                  f"Coarse Acc: {train_coarse_acc:.2f}%, "
                  f"Fine Acc: {train_fine_acc:.2f}%")
            
            # Evaluate
            test_loss, test_coarse_acc, test_fine_acc = evaluate_multihead(
                model, test_loader, criterion, device
            )
            print(f"Test Loss: {test_loss:.4f}, "
                  f"Coarse Acc: {test_coarse_acc:.2f}%, "
                  f"Fine Acc: {test_fine_acc:.2f}%")
            
            # Save best model (using fine accuracy as metric)
            if test_fine_acc > best_acc:
                best_acc = test_fine_acc
                os.makedirs('checkpoints', exist_ok=True)
                torch.save(model.state_dict(), 
                          f'checkpoints/{args.model_type}_{args.backbone}_best.pth')
                print(f"Saved best model with fine accuracy: {best_acc:.2f}%")
        
        scheduler.step()
    
    print(f"\nTraining completed! Best accuracy: {best_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Train CIFAR-100 models')
    parser.add_argument('--model_type', type=str, default='fine',
                        choices=['coarse', 'fine', 'multihead'],
                        help='Type of model to train')
    parser.add_argument('--backbone', type=str, default='mobilenetv3_small_100',
                        help='Backbone architecture from timm')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory to store dataset')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
