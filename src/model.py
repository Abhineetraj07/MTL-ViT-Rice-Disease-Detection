"""
Multi-Task Vision Transformer for Rice Disease and Nutrient Deficiency Detection
"""

import torch
import torch.nn as nn
import timm


class MultiTaskViT(nn.Module):
    """
    Multi-Task Vision Transformer
    
    Architecture:
        - Shared ViT Backbone (pretrained)
        - Disease Classification Head (6 classes)
        - Nutrient Deficiency Head (3 classes)
    """
    
    def __init__(self, num_disease_classes=6, num_nutrient_classes=3, pretrained=True):
        super(MultiTaskViT, self).__init__()
        
        # Load pretrained ViT backbone
        self.backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=0
        )
        
        self.feature_dim = self.backbone.embed_dim  # 768
        
        # Disease classification head
        self.disease_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_disease_classes)
        )
        
        # Nutrient deficiency classification head
        self.nutrient_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_nutrient_classes)
        )
        
        print(f" MultiTaskViT initialized!")
        print(f"   Backbone: ViT-Base-Patch16-224")
        print(f"   Feature dimension: {self.feature_dim}")
        print(f"   Disease head: {self.feature_dim} → 256 → {num_disease_classes} classes")
        print(f"   Nutrient head: {self.feature_dim} → 256 → {num_nutrient_classes} classes")
        
    def forward(self, x):
        features = self.backbone(x)
        disease_logits = self.disease_head(features)
        nutrient_logits = self.nutrient_head(features)
        return disease_logits, nutrient_logits


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
    model = MultiTaskViT(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Test model creation
    model = MultiTaskViT(pretrained=True)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    disease_out, nutrient_out = model(dummy_input)
    print(f"Disease output shape: {disease_out.shape}")
    print(f"Nutrient output shape: {nutrient_out.shape}")
