"""
=============================================================================
MODULE : prepare_dataset.py
RÔLE   : Extraire, standardiser et copier les volumes NRRD et leurs masques
         vers l'arborescence finale pour l'entraînement (Format MSD/nnU-Net).
=============================================================================
"""

import os
import shutil
from pathlib import Path

# --- 1. Configuration des chemins ---
# ⚠️ À MODIFIER : Chemin vers le dossier contenant les 32 patients
SOURCE_DIR = Path(r"C:\Users\Arthur\Documents\Travail\Polytech\4A\Stage\CT Database\SEGMENTARI VENOS DOC 1")

# Chemins de destination (relatifs depuis le dossier python/)
DEST_IMAGES_DIR = Path(r"..\data\03_processed\imagesTr")
DEST_LABELS_DIR = Path(r"..\data\03_processed\labelsTr")

def build_dataset(source_dir: Path, img_dest: Path, lbl_dest: Path):
    print(f"🚀 Début de la préparation du dataset depuis : {source_dir.name}\n")
    
    # S'assurer que les dossiers de destination existent
    img_dest.mkdir(parents=True, exist_ok=True)
    lbl_dest.mkdir(parents=True, exist_ok=True)
    
    patient_count = 1
    
    # Parcourir tous les sous-dossiers (chaque sous-dossier = un patient)
    for folder_path in source_dir.iterdir():
        if not folder_path.is_dir():
            continue
            
        # 1. Identifier le masque (il s'appelle toujours Segmentation.seg.nrrd)
        mask_file = folder_path / "Segmentation.seg.nrrd"
        
        # 2. Identifier l'image (C'est un .nrrd, mais ce n'est PAS le masque)
        image_file = None
        for file in folder_path.glob("*.nrrd"):
            if "Segmentation.seg" not in file.name:
                image_file = file
                break
                
        # Vérification : A-t-on bien trouvé les deux fichiers ?
        if mask_file.exists() and image_file and image_file.exists():
            
            # --- Standardisation des noms ---
            # Format attendu : patient_01, patient_02, etc.
            patient_id = f"patient_{patient_count:03d}" 
            
            # 💡 Bonne pratique (Format nnU-Net) : 
            # On ajoute _0000 à l'image pour indiquer qu'il s'agit de la modalité principale (CT)
            new_image_name = f"{patient_id}_0000.nrrd"
            new_mask_name = f"{patient_id}.nrrd"
            
            dest_image_path = img_dest / new_image_name
            dest_mask_path = lbl_dest / new_mask_name
            
            # --- Copie des fichiers ---
            # shutil.copy2 permet de conserver les métadonnées de création du fichier
            shutil.copy2(image_file, dest_image_path)
            shutil.copy2(mask_file, dest_mask_path)
            
            print(f"✅ [{patient_id}] Traité : {folder_path.name}")
            patient_count += 1
            
        else:
            print(f"⚠️ [{folder_path.name}] Ignoré : Image ou masque introuvable.")

    print(f"\n🎉 Terminé ! {patient_count - 1} patients ont été préparés avec succès.")
    print(f"📂 Images sauvées dans : {img_dest.resolve()}")
    print(f"📂 Masques sauvés dans : {lbl_dest.resolve()}")

# =========================================================================
# EXÉCUTION
# =========================================================================
if __name__ == "__main__":
    if not SOURCE_DIR.exists():
        print(f"❌ Erreur : Le dossier source {SOURCE_DIR} n'existe pas.")
    else:
        build_dataset(SOURCE_DIR, DEST_IMAGES_DIR, DEST_LABELS_DIR)