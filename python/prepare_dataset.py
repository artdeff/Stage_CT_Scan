"""
=============================================================================
MODULE : prepare_dataset.py
RÔLE   : Extraire, standardiser et copier les volumes NRRD et leurs masques
         depuis MULTIPLES dossiers sources vers l'arborescence finale pour 
         l'entraînement (en conservant les noms des patients).
=============================================================================
"""

import os
import shutil
from pathlib import Path

# --- 1. Configuration des chemins ---
# 🌟 NOUVEAU : Une liste contenant tes DEUX dossiers sources
SOURCE_DIRS = [
    Path(r"C:\Users\Arthur\Documents\Travail\Polytech\4A\Stage\CT Database\SEGMENTARI VENOS DOC 1"),
    Path(r"C:\Users\Arthur\Documents\Travail\Polytech\4A\Stage\CT Database\SEGMENTARI PE FILTRE normalized_medlift_gamma[3 3 3]DOC")
]

# Chemins de destination (relatifs depuis le dossier python/)
DEST_IMAGES_DIR = Path(r"..\data\03_processed\imagesTr")
DEST_LABELS_DIR = Path(r"..\data\03_processed\labelsTr")

def build_dataset(source_dirs, img_dest, lbl_dest):
    print("🚀 Début de la préparation du dataset Multi-Dossiers...\n")
    
    # S'assurer que les dossiers de destination existent
    img_dest.mkdir(parents=True, exist_ok=True)
    lbl_dest.mkdir(parents=True, exist_ok=True)
    
    patient_count = 0
    
    # 🌟 NOUVEAU : On boucle sur chacun de tes dossiers sources
    for current_source_dir in source_dirs:
        if not current_source_dir.exists():
            print(f"❌ Erreur : Le dossier source {current_source_dir.name} n'existe pas. Ignoré.")
            continue
            
        print(f"📂 Scan du dossier : {current_source_dir.name}")
        
        # Parcourir tous les sous-dossiers (chaque sous-dossier = un patient)
        for folder_path in current_source_dir.iterdir():
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
                
                # --- Standardisation des noms (SAUVEGARDE DE L'IDENTITÉ) ---
                # On utilise le nom du dossier (ex: "Dragomir Nicolae" -> "Dragomir_Nicolae")
                patient_id = folder_path.name.replace(" ", "_")
                
                # 💡 On ajoute _0000 à l'image pour la modalité principale (CT)
                new_image_name = f"{patient_id}_0000.nrrd"
                new_mask_name = f"{patient_id}.nrrd"
                
                dest_image_path = img_dest / new_image_name
                dest_mask_path = lbl_dest / new_mask_name
                
                # --- Copie des fichiers ---
                shutil.copy2(image_file, dest_image_path)
                shutil.copy2(mask_file, dest_mask_path)
                
                print(f"  ✅ Traité : {patient_id}")
                patient_count += 1
                
            else:
                print(f"  ⚠️ Ignoré [{folder_path.name}] : Image ou masque introuvable.")

    print(f"\n🎉 Terminé ! {patient_count} patients au total ont été préparés avec succès.")
    print(f"📂 Images sauvées dans : {img_dest.resolve()}")
    print(f"📂 Masques sauvés dans : {lbl_dest.resolve()}")

# =========================================================================
# EXÉCUTION
# =========================================================================
if __name__ == "__main__":
    build_dataset(SOURCE_DIRS, DEST_IMAGES_DIR, DEST_LABELS_DIR)