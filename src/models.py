import torch
import torch.nn as nn
import timm


def count_parameters(model):
    """Conta parâmetros treináveis do modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CIFAR100SuperclassClassifier(nn.Module):
    """Classificador de 20 superclasses."""
    def __init__(self, backbone_name="mobilenetv3_small_100", num_classes=20, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        
        # Obtém o número correto de features APÓS criar o modelo
        self.backbone.eval()  # Coloca em eval para evitar erro de BatchNorm
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32)
            num_features = self.backbone(dummy_input).shape[1]
        
        self.classifier = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class CIFAR100FineClassifier(nn.Module):
    """Classificador de 100 classes finas."""
    def __init__(self, backbone_name="mobilenetv3_small_100", num_classes=100, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        
        self.backbone.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32)
            num_features = self.backbone(dummy_input).shape[1]
        
        self.classifier = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class CIFAR100MultiHead(nn.Module):
    """Modelo multi-head: 20 superclasses + 100 classes finas."""
    def __init__(self, backbone_name="mobilenetv3_small_100", num_coarse=20, num_fine=100, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        
        self.backbone.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32)
            num_features = self.backbone(dummy_input).shape[1]
        
        # Duas MLPs separadas (conforme especificação do trabalho)
        # MLP para superclasses
        self.head_coarse = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_coarse)
        )
        
        # MLP para classes finas
        self.head_fine = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_fine)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        out_coarse = self.head_coarse(features)
        out_fine = self.head_fine(features)
        return out_coarse, out_fine


def create_model(task="superclass", backbone="mobilenetv3_small_100", pretrained=False):
    """
    Factory para criar modelos.
    task: 'superclass', 'fine', ou 'multihead'
    """
    if task == "superclass":
        model = CIFAR100SuperclassClassifier(backbone_name=backbone, num_classes=20, pretrained=pretrained)
    elif task == "fine":
        model = CIFAR100FineClassifier(backbone_name=backbone, num_classes=100, pretrained=pretrained)
    elif task == "multihead":
        model = CIFAR100MultiHead(backbone_name=backbone, num_coarse=20, num_fine=100, pretrained=pretrained)
    else:
        raise ValueError(f"task deve ser 'superclass', 'fine' ou 'multihead', recebido: {task}")
    
    return model


if __name__ == "__main__":
    # Testa criação e contagem de parâmetros
    for task in ["superclass", "fine", "multihead"]:
        model = create_model(task=task)
        n_params = count_parameters(model)
        print(f"{task:12s}: {n_params:,} parâmetros")
        
        # Testa forward pass
        x = torch.randn(2, 3, 32, 32)
        model.eval()
        if task == "multihead":
            out_coarse, out_fine = model(x)
            print(f"  out_coarse shape: {out_coarse.shape}, out_fine shape: {out_fine.shape}")
        else:
            out = model(x)
            print(f"  out shape: {out.shape}")