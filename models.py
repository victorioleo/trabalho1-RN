"""
Model definitions for CIFAR-100 classification tasks.

Three model types:
1. Coarse model: 20 superclasses
2. Fine model: 100 classes
3. Multi-head model: both 20 superclasses and 100 classes
"""
import torch
import torch.nn as nn
import timm


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CoarseModel(nn.Module):
    """Model for 20 superclasses classification."""
    
    def __init__(self, model_name='mobilenetv3_small_100', pretrained=False):
        super().__init__()
        # Create base model from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=20
        )
        
    def forward(self, x):
        return self.backbone(x)


class FineModel(nn.Module):
    """Model for 100 classes classification."""
    
    def __init__(self, model_name='mobilenetv3_small_100', pretrained=False):
        super().__init__()
        # Create base model from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=100
        )
        
    def forward(self, x):
        return self.backbone(x)


class MultiHeadModel(nn.Module):
    """
    Multi-head model with two output heads:
    - Coarse head: 20 superclasses
    - Fine head: 100 classes
    
    The final loss is the sum of losses from both heads.
    """
    
    def __init__(self, model_name='mobilenetv3_small_100', pretrained=False):
        super().__init__()
        # Create base model without classification head
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove the classification head
        )
        
        # Get the number of features from the backbone
        # Use a forward pass to determine the actual output size
        self.backbone.eval()  # Set to eval mode to avoid batch norm issues
        with torch.no_grad():
            dummy_input = torch.randn(2, 3, 32, 32)  # Use batch size 2
            dummy_output = self.backbone(dummy_input)
            num_features = dummy_output.shape[1]
        self.backbone.train()  # Set back to train mode
        
        # Create two classification heads
        self.coarse_head = nn.Linear(num_features, 20)
        self.fine_head = nn.Linear(num_features, 100)
        
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Apply both heads
        coarse_output = self.coarse_head(features)
        fine_output = self.fine_head(features)
        
        return coarse_output, fine_output


def create_model(model_type='coarse', model_name='mobilenetv3_small_100', pretrained=False):
    """
    Create a model based on the specified type.
    
    Args:
        model_type: One of 'coarse', 'fine', or 'multihead'
        model_name: Name of the timm model to use as backbone
        pretrained: Whether to use pretrained weights
        
    Returns:
        model: The created model
    """
    if model_type == 'coarse':
        model = CoarseModel(model_name=model_name, pretrained=pretrained)
    elif model_type == 'fine':
        model = FineModel(model_name=model_name, pretrained=pretrained)
    elif model_type == 'multihead':
        model = MultiHeadModel(model_name=model_name, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'coarse', 'fine', or 'multihead'")
    
    num_params = count_parameters(model)
    print(f"Created {model_type} model with {num_params:,} parameters")
    
    if num_params > 10_000_000:
        print(f"WARNING: Model has {num_params:,} parameters, which exceeds the 10M limit!")
    
    return model


if __name__ == '__main__':
    # Test model creation
    print("Testing model creation...")
    
    print("\n1. Coarse Model (20 classes):")
    coarse_model = create_model('coarse')
    x = torch.randn(2, 3, 32, 32)
    out = coarse_model(x)
    print(f"   Input shape: {x.shape}, Output shape: {out.shape}")
    
    print("\n2. Fine Model (100 classes):")
    fine_model = create_model('fine')
    out = fine_model(x)
    print(f"   Input shape: {x.shape}, Output shape: {out.shape}")
    
    print("\n3. Multi-head Model (20 + 100 classes):")
    multihead_model = create_model('multihead')
    coarse_out, fine_out = multihead_model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Coarse output shape: {coarse_out.shape}")
    print(f"   Fine output shape: {fine_out.shape}")
