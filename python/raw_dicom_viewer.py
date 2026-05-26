"""
=============================================================================
MODULE : raw_dicom_viewer.py
RÔLE   : Explorer interactivement les dossiers de scanner DICOM bruts,
         reconstituer le volume 3D et afficher une coupe de contrôle.
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from monai.transforms import LoadImage

# Configuration du chemin racine des données brutes
RAW_DATA_DIR = Path(r"..\data\01_raw")

def select_dicom_folder_interactive(base_dir: Path) -> Path | None:
    """Menu interactif pour choisir le patient puis la série de coupes axiales."""
    if not base_dir.exists():
        print(f"❌ Erreur : Le dossier racine {base_dir.resolve()} n'existe pas.")
        return None

    # 1. Sélection du Patient
    patients = [d for d in base_dir.iterdir() if d.is_dir()]
    if not patients:
        print(f"⚠️ Aucun sous-dossier trouvé dans {base_dir.name}")
        return None

    print("\n" + "="*60)
    print("👤 LISTE DES PATIENTS BRUTS DISPONIBLES :")
    print("="*60)
    for i, p in enumerate(patients):
        print(f"  [{i}] -> {p.name}")
    print("="*60)

    while True:
        try:
            choix_p = input("\n👉 Choisis le numéro du PATIENT (ex: 0) : ")
            idx_p = int(choix_p)
            if 0 <= idx_p < len(patients):
                patient_dir = patients[idx_p]
                break
            print(f"❌ Numéro invalide. Choisis entre 0 et {len(patients)-1}.")
        except ValueError:
            print("❌ Entre un nombre entier valide.")

    # 2. Sélection de la Série (Sous-dossier)
    series = [d for d in patient_dir.iterdir() if d.is_dir()]
    if not series:
        print(f"⚠️ Aucun sous-dossier de série trouvé pour le patient {patient_dir.name}")
        return None

    print("\n" + "="*60)
    print(f"📂 SÉRIES (PHASES) DISPONIBLES POUR LE PATIENT {patient_dir.name} :")
    print("="*60)
    for i, s in enumerate(series):
        # Petit conseil visuel à l'utilisateur
        conseil = ""
        if "AX" in s.name.upper():
            conseil = " 🌟 (Recommandé - Vue Axiale)"
        elif "COR" in s.name.upper() or "SAG" in s.name.upper():
            conseil = " ⚠️ (Attention - Pas le bon plan pour l'IA)"
        print(f"  [{i}] -> {s.name}{conseil}")
    print("="*60)

    while True:
        try:
            choix_s = input(f"\n👉 Choisis le numéro de la SÉRIE à visualiser (ex: 0) : ")
            idx_s = int(choix_s)
            if 0 <= idx_s < len(series):
                return series[idx_s]
            print(f"❌ Numéro invalide. Choisis entre 0 et {len(series)-1}.")
        except ValueError:
            print("❌ Entre un nombre entier valide.")

def view_raw_dicom_series():
    # Appel du menu de sélection
    dossier_cible = select_dicom_folder_interactive(RAW_DATA_DIR)
    
    if dossier_cible is None:
        print("❌ Opération annulée ou impossible.")
        return

    print(f"\n⏳ Chargement et reconstruction 3D de la série : {dossier_cible.name}...")
    
    # Initialisation du chargeur ITK de MONAI
    loader = LoadImage(reader="ITKReader", image_only=False)
    
    try:
        # ITK lit le dossier, trie les fichiers .dcm par instance/numéro et crée le volume
        image_vol, meta_data = loader(str(dossier_cible))
    except Exception as e:
        print(f"\n❌ Erreur de lecture : Ce dossier ne contient probablement pas de fichiers .dcm valides.")
        print(f"Détails de l'erreur : {e}")
        return
        
    image_vol = np.array(image_vol)
    
    print("\n✅ Reconstruction 3D réussie !")
    print(f"📏 Dimensions spatiales de la matrice (X, Y, Z) : {image_vol.shape}")
    
    # Extraction de la coupe centrale (Axe Z)
    mid_z = image_vol.shape[2] // 2
    
    # Configuration de l'affichage
    plt.figure(figsize=(8, 8))
    # Fenêtrage Tissus Mous standard (-150 à 250 Hounsfield Units)
    plt.imshow(image_vol[:, :, mid_z].T, cmap='gray', vmin=-150, vmax=250)
    plt.title(f"Visualisation de Contrôle (Coupe Z={mid_z})\nPatient: {dossier_cible.parent.name} | Série: {dossier_cible.name}")
    plt.axis('off')
    
    print("🖼️ Ouverture de la fenêtre graphique Matplotlib...")
    plt.show()

# =========================================================================
# EXÉCUTION
# =========================================================================
if __name__ == "__main__":
    view_raw_dicom_series()