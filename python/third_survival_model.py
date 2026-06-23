"""
=============================================================================
MODULE : third_survival_model.py
RÔLE   : Définir la troisième architecture (ResNet10 3D) pour le vote majoritaire.
=============================================================================
"""

import torch
from monai.networks.nets import ResNet, resnet10

def create_resnet_survival_classifier(spatial_dims=3, n_input_channels=1, num_classes=2):
    """
    Instancie un modèle ResNet-10 3D adapté pour la classification de survie.
    Utilise des blocs résiduels de base pour une extraction stable des features.
    """
    # resnet10 de MONAI configuré pour des volumes 3D
    model = resnet10(
        spatial_dims=spatial_dims,
        n_input_channels=n_input_channels,
        num_classes=num_classes,
        feed_forward=True
    )
    return model

if __name__ == "__main__":
    print("⏳ Test de l'architecture ResNet3D...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_resnet_survival_classifier().to(device)
    
    # Simulation d'un batch de 2 cubes (64x64x64)
    dummy_input = torch.randn(2, 1, 64, 64, 64).to(device)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        
    print("✅ Modèle ResNet3D opérationnel !")
    print(f"📐 Forme du tenseur de sortie : {output.shape}") # (2, 2)