"""
=============================================================================
MODULE : alternative_survival_model.py
RÔLE   : Définir l'architecture alternative (EfficientNet3D) pour la 
         prédiction de survie afin de croiser les résultats avec DenseNet3D.
=============================================================================
"""

import torch
import torch.nn as nn
from monai.networks.nets import EfficientNetBN

def create_alternative_survival_classifier(spatial_dims=3, ina_channels=1, num_classes=2):
    """
    Instancie un modèle EfficientNet-B0 3D adapté pour la classification.
    Ce modèle utilise l'autocalibrage des blocs de convolution pour être
    très précis sur les textures hétérogènes.
    """
    # On utilise la version EfficientNet-B0 de MONAI adaptée à la 3D
    model = EfficientNetBN(
        model_name="efficientnet-b0",
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        num_classes=num_classes
    )
    
    return model

if __name__ == "__main__":
    # Petit test à blanc (Dummy Test) pour vérifier que les tenseurs circulent bien
    print("⏳ Test de l'architecture alternative EfficientNet3D...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = create_alternative_survival_classifier(spatial_dims=3, in_channels=1, num_classes=2).to(device)
    
    # Simulation d'un batch de 2 cubes de pancréas (Taille 64x64x64)
    dummy_input = torch.randn(2, 1, 64, 64, 64).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    print("✅ Modèle alternatif opérationnel !")
    print(f"📐 Forme du tenseur de sortie (Batch, Classes) : {output.shape}") # Doit être (2, 2)