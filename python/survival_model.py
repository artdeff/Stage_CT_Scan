"""
=============================================================================
MODULE : survival_model.py
RÔLE   : Définir l'architecture du modèle d'IA 3D pour la prédiction 
         de la survie à partir des cubes de pancréas.
=============================================================================
"""

import torch
import torch.nn as nn
from monai.networks.nets import densenet121

def create_survival_classifier(spatial_dims=3, in_channels=1, num_classes=2):
    """
    Initialise un DenseNet121 3D pour la classification de la survie.
    Prend en entrée un cube 3D de taille variable (ex: 40x50x35) et 
    sort des probabilités pour 'num_classes' (Survie Courte vs Longue).
    """
    # Chargement du DenseNet121 configuré en 3D par MONAI
    model_3d = densenet121(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=num_classes # Sortie : 2 classes (logits)
    )
    
    return model_3d

if __name__ == "__main__":
    # Petit test de validation de l'architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_survival_classifier().to(device)
    
    # Simulation d'un faux lot (batch) de 2 cubes de pancréas de taille 64x64x64
    fake_batch = torch.randn(2, 1, 64, 64, 64).to(device)
    
    with torch.no_grad():
        output = model(fake_batch)
    print(f"✅ Modèle DenseNet3D initialisé avec succès !")
    print(f"📐 Forme de la sortie (Batch, Classes) : {output.shape} -> Doit être (2, 2)")