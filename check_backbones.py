"""
Script to check parameter counts for different backbone architectures.

This helps identify which models are suitable (< 10M parameters).
"""
import timm
from models import create_model, count_parameters


def check_backbone(model_name, description=""):
    """Check if a backbone model meets the parameter constraint."""
    print(f"\nTesting: {model_name}")
    if description:
        print(f"  Description: {description}")
    
    try:
        # Try each model type
        for model_type in ['coarse', 'fine', 'multihead']:
            model = create_model(model_type, model_name=model_name, pretrained=False)
            params = count_parameters(model)
            status = "✓" if params <= 10_000_000 else "✗ (too large)"
            print(f"  {model_type:10s}: {params:>10,} params {status}")
            del model  # Free memory
    except Exception as e:
        print(f"  ERROR: {e}")


def main():
    print("=" * 70)
    print("CIFAR-100 Backbone Architecture Comparison")
    print("=" * 70)
    print("\nChecking parameter counts for different backbones...")
    print("Requirement: Models must have < 10M parameters")
    
    # Test suitable models (< 10M params)
    print("\n" + "=" * 70)
    print("RECOMMENDED MODELS (< 10M parameters)")
    print("=" * 70)
    
    check_backbone('mobilenetv3_small_100', 'Lightweight MobileNetV3')
    check_backbone('mobilenetv3_large_100', 'Larger MobileNetV3')
    check_backbone('efficientnet_b0', 'EfficientNet B0')
    
    # Test models that might be too large
    print("\n" + "=" * 70)
    print("MODELS TO VERIFY (might exceed 10M)")
    print("=" * 70)
    
    check_backbone('resnet18', 'ResNet-18')
    check_backbone('resnet34', 'ResNet-34')
    
    # Additional lightweight options
    print("\n" + "=" * 70)
    print("OTHER LIGHTWEIGHT OPTIONS")
    print("=" * 70)
    
    try:
        check_backbone('mobilevitv2_050', 'MobileViT v2 (50% width)')
    except:
        print("\nmobilevitv2_050: Not available in this timm version")
    
    try:
        check_backbone('tf_efficientnet_lite0', 'EfficientNet Lite 0')
    except:
        print("\ntf_efficientnet_lite0: Not available or error")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nRecommended backbones for this project:")
    print("  1. mobilenetv3_small_100  (~1.5-1.6M params) - Default, very lightweight")
    print("  2. mobilenetv3_large_100  (~5.5-5.6M params) - Good balance")
    print("  3. efficientnet_b0        (~5.3-5.4M params) - Efficient architecture")
    print("\nTo use a different backbone:")
    print("  python train.py --model_type multihead --backbone mobilenetv3_large_100")
    print("=" * 70)


if __name__ == '__main__':
    main()
