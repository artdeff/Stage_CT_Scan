"""
=============================================================================
MODULE : extract_pancreas_cube.py
RÔLE   : Extraire la boîte englobante (bounding box) du pancréas à partir
         des images filtrées et des masques pour préparer l'IA de survie.
=============================================================================
"""

import os
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, ScaleIntensityRanged

# Configuration des chemins
DATA_DIR = Path(r"..\data\03_processed")
IMAGES_DIR = DATA_DIR / "imagesTr"
LABELS_DIR = DATA_DIR / "labelsTr"

# Dossier de sortie pour les cubes de pancréas purifiés
CUBES_OUTPUT_DIR = DATA_DIR / "pancreas_cubes"
CUBES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_bounding_box_3d(image_np, label_np, margin=5):
    """Calcule les coordonnées de la boîte englobante autour du pancréas (label > 0)."""
    # On cherche tous les indices où le pancréas ou la tumeur sont présents
    indices = np.argwhere(label_np > 0)
    
    if indices.size == 0:
        return None # Aucun pancréas trouvé dans ce volume
        
    # Bornes minimales et maximales sur les 3 axes (C, H, W, D) -> on ignore le canal C (axe 0)
    min_h, min_w, min_d = indices[:, 1].min(), indices[:, 2].min(), indices[:, 3].min()
    max_h, max_w, max_d = indices[:, 1].max(), indices[:, 2].max(), indices[:, 3].max()
    
    # Ajout d'une petite marge de sécurité pour ne pas couper au ras de l'organe
    H, W, D = image_np.shape[1:]
    min_h = max(0, min_h - margin)
    min_w = max(0, min_w - margin)
    min_d = max(0, min_d - margin)
    
    max_h = min(H, max_h + margin)
    max_w = min(W, max_w + margin)
    max_d = min(D, max_d + margin)
    
    # Découpage du cube (on garde le canal complet)
    cropped_image = image_np[:, min_h:max_h, min_w:max_w, min_d:max_d]
    return cropped_image

def pipeline_extraction_cubes():
    print("✂️ Lancement de la pipeline d'extraction des régions pancréatiques...")
    
    image_files = sorted(list(IMAGES_DIR.glob("*.nrrd")))
    label_files = sorted(list(LABELS_DIR.glob("*.nrrd")))
    
    # Pipeline de prétraitement/filtrage (Étape 1 et 2 demandées)
    preprocessing = Compose([
        LoadImaged(keys=["image", "label"], reader="ITKReader"),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-150, a_max=250, b_min=0.0, b_max=1.0, clip=True),
    ])
    
    count = 0
    for img_path, lbl_path in zip(image_files, label_files):
        print(f"🔄 Traitement du patient : {img_path.name}")
        
        # 1. Appliquer le prétraitement/filtrage sur l'image et le masque
        data = preprocessing({"image": str(img_path), "label": str(lbl_path)})
        
        img_np = data["image"].numpy()
        lbl_np = data["label"].numpy()
        
        # 2. Extraire la Bounding Box du pancréas à partir de l'image filtrée
        pancreas_cube = extract_bounding_box_3d(img_np, lbl_np, margin=4)
        
        if pancreas_cube is not None:
            # Sauvegarde du cube en .nrrd
            output_path = CUBES_OUTPUT_DIR / f"cube_{img_path.name}"
            
            # SimpleITK attend (D, W, H) à la place de (C, H, W, D)
            # On enlève la dimension canal [0] car le cube est en niveaux de gris
            sitk_img = sitk.GetImageFromArray(pancreas_cube[0].transpose(2, 1, 0))
            
            # Sauvegarde
            sitk.WriteImage(sitk_img, str(output_path))
            print(f"  ✅ Cube extrait avec succès ({pancreas_cube.shape[1:]}) -> {output_path.name}")
            count += 1
        else:
            print(f"  ⚠️ Attention : Aucun pixel de pancréas détecté pour ce patient.")
            
    print(f"\n🎉 Extraction terminée ! {count} cubes de pancréas standardisés sont sauvegardés dans : {CUBES_OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    pipeline_extraction_cubes()