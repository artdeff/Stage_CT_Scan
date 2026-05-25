"""
=============================================================================
MODULE : unet_segmentation.py
RÔLE   : Définir l'architecture U-Net (MONAI) pour la segmentation.
=============================================================================
"""

import torch
from monai.networks.nets import UNet

def create_unet_model(spatial_dims: int = 3, in_channels: int = 1, out_channels: int = 4):
    """
    Initialise un modèle U-Net robuste via MONAI.
    
    Args:
        spatial_dims: 2 pour des coupes 2D, 3 pour des volumes 3D.
        in_channels: Nombre de canaux en entrée (1 pour du CT scan en niveaux de gris).
        out_channels: Nombre de classes en sortie (ex: 2 pour 'Fond' vs 'Pancréas').
        
    Returns:
        Modèle PyTorch U-Net.
    """
    model = UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),  # Nombre de filtres par couche
        strides=(2, 2, 2, 2),             # Réduction de dimension spatiale
        num_res_units=2,                  # Blocs résiduels pour un meilleur apprentissage
        norm="batch"                      # Normalisation
    )
    return model

# =========================================================================
# TEST À BLANC (Dummy Test)
# =========================================================================
if __name__ == "__main__":
    print("🧠 Initialisation du modèle U-Net (MONAI)...")
    
    # Création d'un modèle 3D pour 1 organe cible + le fond (2 classes)
    model = create_unet_model(spatial_dims=3, in_channels=1, out_channels=2)
    
    # Création d'un "faux" tenseur représentant un petit patch 3D de scanner
    # Format attendu par PyTorch : (Batch, Channel, Z, Y, X)
    # Exemple : 1 volume, 1 canal (niveaux de gris), patch de taille 64x64x64
    dummy_input = torch.randn(1, 1, 64, 64, 64)
    
    print(f"📦 Forme de l'entrée (CT Patch) : {dummy_input.shape}")
    
    # Inférence à blanc : on fait passer nos fausses données dans le réseau
    output = model(dummy_input)
    
    print(f"🎯 Forme de la sortie (Masque prédit) : {output.shape}")
    print("✅ Modèle U-Net prêt et fonctionnel !")