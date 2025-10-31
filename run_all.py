"""
Script principal para executar todos os experimentos do Trabalho 1.
Treina os 3 modelos (superclass, fine, multihead) e avalia cada um.
"""
import os
import sys
import subprocess
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import timm


def run_command(cmd, description):
    """Executa um comando e exibe o progresso."""
    print("\n" + "="*80)
    print(f"► {description}")
    print("="*80)
    print(f"Comando: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n✗ ERRO ao executar: {description}")
        sys.exit(1)
    else:
        print(f"\n✓ Concluído: {description}")


def main(args):
    """Executa todos os experimentos do trabalho."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = args.output_dir or f"runs/experiment_{timestamp}"
    
    print("="*80)
    print("TRABALHO 1 - MODELO MULTI-HEAD")
    print("CIFAR-100: Treinamento de 3 modelos")
    print("="*80)
    print(f"Diretório de saída: {base_output_dir}")
    print(f"Device: {'CUDA' if args.use_cuda else 'CPU'}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Patience: {args.patience}")
    print("="*80)
    
    # Configurações comuns
    common_args = [
        "--data-root", args.data_root,
        "--val-size", str(args.val_size),
        "--seed", str(args.seed),
        "--backbone", args.backbone,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--patience", str(args.patience),
        "--num-workers", str(args.num_workers),
    ]
    
    if args.pretrained:
        common_args.append("--pretrained")
    
    # ==========================================
    # 1. MODELO DE 20 SUPERCLASSES
    # ==========================================
    experiment = "superclass"
    output_dir = os.path.join(base_output_dir, experiment)
    
    train_cmd = [
        sys.executable, "src/train.py",
        "--task", experiment,
        "--output-dir", output_dir,
    ] + common_args
    
    run_command(train_cmd, f"1/3 - Treinando modelo de 20 SUPERCLASSES")
    
    # Avaliação
    checkpoint_path = os.path.join(output_dir, f"best_model_{experiment}.pth")
    report_path = os.path.join(output_dir, "classification_report.txt")
    
    eval_cmd = [
        sys.executable, "src/eval.py",
        "--task", experiment,
        "--backbone", args.backbone,
        "--checkpoint", checkpoint_path,
        "--data-root", args.data_root,
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--output-file", report_path,
    ]
    
    run_command(eval_cmd, "Avaliando modelo de 20 SUPERCLASSES")
    
    # ==========================================
    # 2. MODELO DE 100 CLASSES FINAS
    # ==========================================
    experiment = "fine"
    output_dir = os.path.join(base_output_dir, experiment)
    
    train_cmd = [
        sys.executable, "src/train.py",
        "--task", experiment,
        "--output-dir", output_dir,
    ] + common_args
    
    run_command(train_cmd, f"2/3 - Treinando modelo de 100 CLASSES FINAS")
    
    # Avaliação
    checkpoint_path = os.path.join(output_dir, f"best_model_{experiment}.pth")
    report_path = os.path.join(output_dir, "classification_report.txt")
    
    eval_cmd = [
        sys.executable, "src/eval.py",
        "--task", experiment,
        "--backbone", args.backbone,
        "--checkpoint", checkpoint_path,
        "--data-root", args.data_root,
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--output-file", report_path,
    ]
    
    run_command(eval_cmd, "Avaliando modelo de 100 CLASSES FINAS")
    
    # ==========================================
    # 3. MODELO MULTI-HEAD
    # ==========================================
    experiment = "multihead"
    output_dir = os.path.join(base_output_dir, experiment)
    
    train_cmd = [
        sys.executable, "src/train.py",
        "--task", experiment,
        "--output-dir", output_dir,
    ] + common_args
    
    run_command(train_cmd, f"3/3 - Treinando modelo MULTI-HEAD (20 + 100)")
    
    # Avaliação
    checkpoint_path = os.path.join(output_dir, f"best_model_{experiment}.pth")
    report_path = os.path.join(output_dir, "classification_report.txt")
    
    eval_cmd = [
        sys.executable, "src/eval.py",
        "--task", experiment,
        "--backbone", args.backbone,
        "--checkpoint", checkpoint_path,
        "--data-root", args.data_root,
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--output-file", report_path,
    ]
    
    run_command(eval_cmd, "Avaliando modelo MULTI-HEAD")
    
    # ==========================================
    # SUMÁRIO FINAL
    # ==========================================
    print("\n" + "="*80)
    print("✓ TODOS OS EXPERIMENTOS CONCLUÍDOS COM SUCESSO!")
    print("="*80)
    print(f"\nResultados salvos em: {base_output_dir}")
    print("\nEstrutura de arquivos:")
    print(f"  {base_output_dir}/")
    print(f"    ├── superclass/")
    print(f"    │   ├── best_model_superclass.pth")
    print(f"    │   ├── loss_curves_superclass.png")
    print(f"    │   └── classification_report.txt")
    print(f"    ├── fine/")
    print(f"    │   ├── best_model_fine.pth")
    print(f"    │   ├── loss_curves_fine.png")
    print(f"    │   └── classification_report.txt")
    print(f"    └── multihead/")
    print(f"        ├── best_model_multihead.pth")
    print(f"        ├── loss_curves_multihead.png")
    print(f"        └── classification_report.txt")
    print("\nPara visualizar os relatórios:")
    for exp in ["superclass", "fine", "multihead"]:
        report = os.path.join(base_output_dir, exp, "classification_report.txt")
        print(f"  type {report}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Executa todos os experimentos do Trabalho 1 automaticamente"
    )
    
    # Dados
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Diretório raiz dos dados CIFAR-100 (default: data)"
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=5000,
        help="Tamanho do conjunto de validação (default: 5000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente para reprodutibilidade (default: 42)"
    )
    
    # Modelo
    parser.add_argument(
        "--backbone",
        type=str,
        default="mobilenetv3_small_100",
        help="Backbone do timm (default: mobilenetv3_small_100)"
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Usar pesos pré-treinados do ImageNet"
    )
    
    # Treinamento
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Número máximo de épocas (default: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size (default: 128)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (default: 1e-4)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Patience para early stopping (default: 15)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Num workers para DataLoader (default: 4)"
    )
    
    # Saída
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Diretório de saída (default: runs/experiment_<timestamp>)"
    )
    
    # Outros
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="Flag informativa (CUDA é detectado automaticamente)"
    )
    
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\n✗ Execução interrompida pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)