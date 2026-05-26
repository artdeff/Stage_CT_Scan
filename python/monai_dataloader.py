"""
=============================================================================
MODULE : monai_dataloader.py
RÔLE   : Pipeline de chargement, transformation 3D et création des batchs.
=============================================================================
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import torch

# Importation de l'artillerie lourde de MONAI
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld, ToTensord,
    Lambdad, RandRotated, RandZoomd, RandGaussianNoised, RandAdjustContrastd # 🌟 Ajouts pour l'augmentation
)
from monai.data import Dataset, DataLoader
from monai.utils import set_determinism

from monai.transforms import Lambdad # 🌟 Ajoute cet import tout en haut !

# Fixer l'aléatoire pour que tes résultats soient reproductibles
set_determinism(seed=42)

# --- 1. Paramètres ---
DATA_DIR = Path(r"..\data\03_processed")
IMAGES_DIR = DATA_DIR / "imagesTr"
LABELS_DIR = DATA_DIR / "labelsTr"

def get_train_transforms():
    """
    Définit la chaîne d'opérations mathématiques appliquées "à la volée"
    sur chaque patient avant qu'il n'entre dans le réseau de neurones.
    """
    return Compose([
        # 1. Chargement de l'image et du masque
        LoadImaged(keys=["image", "label"], reader="ITKReader"),
        
        # 2. Ajout de la dimension "Canal" (H,W,D) -> (C,H,W,D) indispensable pour PyTorch
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # 🌟 NOUVEAU : Nettoyage ABSOLU des labels (Juste après EnsureChannelFirstd)
        # On force toutes les valeurs à être des entiers stricts bridés entre 0 et 3.
        # Cela élimine instantanément les indices hors limites (ex: 4 ou plus).
        Lambdad(
            keys=["label"],
            func=lambda x: torch.clamp(x.round().long(), min=0, max=3)
        ),
        
        # 3. Standardisation de l'orientation du patient (RAS = Right, Anterior, Superior)
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        
        # 4. Standardisation de la taille des pixels physiques (Voxel Spacing)
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        
        # 5. Fenêtrage (Windowing) : On cible les tissus mous/pancréas (HU entre -150 et 250)
        # Et on normalise entre 0.0 et 1.0 pour le réseau de neurones
        ScaleIntensityRanged(
            keys=["image"], a_min=-150, a_max=250, b_min=0.0, b_max=1.0, clip=True
        ),
        
        # 6. On coupe le vide inutile autour du corps du patient
        CropForegroundd(keys=["image", "label"], source_key="image"),
        
        # 7. extraction du patch 3D (ex: 96x96x96)
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=2, # Il va extraire 2 petits cubes par patient
            image_key="image",
            image_threshold=0,
        ),

        # --- 🌟 NOUVEAU : DATA AUGMENTATION ---
        # 1. Rotation aléatoire (Simule un patient mal positionné dans le scanner)
        RandRotated(keys=["image", "label"], range_x=0.3, range_y=0.3, range_z=0.3, prob=0.5, mode=("bilinear", "nearest")),
        # 2. Zoom aléatoire (Simule un patient plus grand ou plus petit)
        RandZoomd(keys=["image", "label"], prob=0.5, min_zoom=0.8, max_zoom=1.2, mode=("bilinear", "nearest")),
        # 3. Bruit Gaussien (Simule un scanner de moins bonne qualité / basse dose)
        RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.1),
        # 4. Variation de contraste (Oblige l'IA à analyser la texture plutôt que la couleur absolue)
        RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.5, 2.0)),
        # --------------------------------------
        
        # 8. Conversion finale en Tenseur PyTorch
        ToTensord(keys=["image", "label"])
    ])

def prepare_dataloaders():
    # Création de la liste des dictionnaires pour MONAI
    # Format attendu : [{"image": "path_img_1", "label": "path_lbl_1"}, ...]
    data_dicts = []
    
    # On récupère tous les fichiers images triés
    image_files = sorted(list(IMAGES_DIR.glob("*.nrrd")))
    label_files = sorted(list(LABELS_DIR.glob("*.nrrd")))
    
    for img_path, lbl_path in zip(image_files, label_files):
        data_dicts.append({"image": str(img_path), "label": str(lbl_path)})
        
    print(f"📦 Total de patients trouvés : {len(data_dicts)}")
    
    # Simple séparation : on garde 26 patients pour l'entraînement, et 6 pour le test
    train_files, val_files = data_dicts[:26], data_dicts[26:]
    
    # Création des Datasets
    train_ds = Dataset(data=train_files, transform=get_train_transforms())
    
    # Création des DataLoaders (batch_size=2 veut dire qu'on envoie 2 cubes à la fois à la carte graphique)
    # num_workers=0 est très important sous Windows pour éviter que le CPU ne crashe au chargement
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    
    return train_loader

# =========================================================================
# TEST DU DATALOADER
# =========================================================================
if __name__ == "__main__":
    print("⏳ Préparation du DataLoader et découpage des premiers patchs 3D...")
    train_loader = prepare_dataloaders()
    
    # On tire le tout premier "batch" (lot) généré par notre pipeline
    batch = next(iter(train_loader))
    
    images = batch["image"]
    labels = batch["label"]
    
    print("\n✅ Batch chargé avec succès !")
    print(f"📏 Forme du Tenseur Image : {images.shape} -> (Batch, Canal, X, Y, Z)")
    print(f"📏 Forme du Tenseur Masque : {labels.shape}")