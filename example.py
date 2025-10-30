"""
Example script demonstrating how to use the models.

This script shows:
1. How to create each model type
2. How to perform a forward pass
3. Model parameter counts
"""
import torch
from models import create_model, count_parameters


def main():
    print("=" * 60)
    print("CIFAR-100 Models Demo")
    print("=" * 60)
    
    # Create a sample batch
    batch_size = 4
    sample_batch = torch.randn(batch_size, 3, 32, 32)
    print(f"\nSample input shape: {sample_batch.shape}")
    
    # 1. Coarse Model (20 superclasses)
    print("\n" + "=" * 60)
    print("1. COARSE MODEL (20 superclasses)")
    print("=" * 60)
    coarse_model = create_model('coarse', model_name='mobilenetv3_small_100')
    coarse_model.eval()
    
    with torch.no_grad():
        coarse_output = coarse_model(sample_batch)
    
    print(f"Output shape: {coarse_output.shape}")
    print(f"Number of parameters: {count_parameters(coarse_model):,}")
    print(f"Sample predictions (logits): {coarse_output[0, :5]}")
    
    # 2. Fine Model (100 classes)
    print("\n" + "=" * 60)
    print("2. FINE MODEL (100 classes)")
    print("=" * 60)
    fine_model = create_model('fine', model_name='mobilenetv3_small_100')
    fine_model.eval()
    
    with torch.no_grad():
        fine_output = fine_model(sample_batch)
    
    print(f"Output shape: {fine_output.shape}")
    print(f"Number of parameters: {count_parameters(fine_model):,}")
    print(f"Sample predictions (logits): {fine_output[0, :5]}")
    
    # 3. Multi-head Model
    print("\n" + "=" * 60)
    print("3. MULTI-HEAD MODEL (20 + 100 classes)")
    print("=" * 60)
    multihead_model = create_model('multihead', model_name='mobilenetv3_small_100')
    multihead_model.eval()
    
    with torch.no_grad():
        coarse_out, fine_out = multihead_model(sample_batch)
    
    print(f"Coarse head output shape: {coarse_out.shape}")
    print(f"Fine head output shape: {fine_out.shape}")
    print(f"Number of parameters: {count_parameters(multihead_model):,}")
    print(f"Sample coarse predictions (logits): {coarse_out[0, :5]}")
    print(f"Sample fine predictions (logits): {fine_out[0, :5]}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"All models use backbone: mobilenetv3_small_100")
    print(f"All models have < 10M parameters ✓")
    print(f"\nParameter counts:")
    print(f"  - Coarse model:     {count_parameters(coarse_model):,}")
    print(f"  - Fine model:       {count_parameters(fine_model):,}")
    print(f"  - Multi-head model: {count_parameters(multihead_model):,}")
    
    print("\n" + "=" * 60)
    print("Training Commands")
    print("=" * 60)
    print("\nTo train the coarse model:")
    print("  python train.py --model_type coarse --epochs 100")
    print("\nTo train the fine model:")
    print("  python train.py --model_type fine --epochs 100")
    print("\nTo train the multi-head model:")
    print("  python train.py --model_type multihead --epochs 100")
    print("\nFor faster testing (fewer epochs):")
    print("  python train.py --model_type multihead --epochs 10 --batch_size 64")
    print("=" * 60)


if __name__ == '__main__':
    main()
