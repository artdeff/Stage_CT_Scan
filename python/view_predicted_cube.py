"""
=============================================================================
MODULE : view_predicted_cube.py (Version Sécurisée SimpleITK)
RÔLE   : Menu interactif pour charger un cube de pancréas extrait (.nrrd)
         via SimpleITK pour contourner les restrictions de sécurité Windows.
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import SimpleITK as sitk  # <-- On utilise SimpleITK directement à la place de MONAI

# 1. Configuration du dossier cible des nouveaux cubes
CUBES_DIR = Path(r"..\data\05_predicted_cubes")

def interactive_predicted_cube_viewer():
    print("\n" + "="*50)
    print("🔍 CUBES D'IA EXTRAITS DISPONIBLES POUR CONTRÔLE (Version Sécurisée) :")
    print("="*50)
    
    cubes = sorted(list(CUBES_DIR.glob("cube_*.nrrd")))
    if not cubes:
        print(f"❌ Aucun cube trouvé dans {CUBES_DIR.resolve()} !")
        return
        
    for i, c in enumerate(cubes):
        patient_id = c.stem.replace("cube_raw_", "")
        print(f"  [{i}] -> Patient : {patient_id}")
    print("="*50)
    
    while True:
        try:
            choix = input("\n👉 Choisis le numéro du cube à visualiser (ex: 0) : ")
            idx = int(choix)
            if 0 <= idx < len(cubes):
                selected_cube = cubes[idx]
                patient_id = selected_cube.stem.replace("cube_raw_", "")
                break
            print(f"❌ Numéro invalide.")
        except ValueError:
            print("❌ Entre un nombre entier.")
            
    print(f"\n⏳ Chargement sécurisé via SimpleITK pour {patient_id}...")
    
    try:
        # 2. Lecture du fichier .nrrd avec SimpleITK
        sitk_img = sitk.ReadImage(str(selected_cube))
        
        # SimpleITK charge en (Z, Y, X), on transpose en (X, Y, Z) pour rester cohérent avec tes habitudes
        cube_array = sitk.GetArrayFromImage(sitk_img).transpose(2, 1, 0)
        
        print(f"📐 Dimensions de ce cube de pancréas (X, Y, Z) : {cube_array.shape}")
        
        # 3. Extraction de la coupe du milieu sur l'axe Z
        mid_z = cube_array.shape[2] // 2
        img_slice = cube_array[:, :, mid_z]
        
        # 4. Affichage graphique (Matplotlib)
        plt.figure(figsize=(7, 7))
        
        # Les valeurs étant déjà normalisées entre 0.0 et 1.0 par l'extracteur
        plt.imshow(img_slice.T, cmap="gray", vmin=0.0, vmax=1.0)
        plt.colorbar(label="Intensité normalisée (0.0 à 1.0)")
        
        plt.title(f"Coupe centrale Z={mid_z} du Pancréas IA\nPatient : {patient_id}", fontsize=12, fontweight='bold')
        plt.axis("off")
        
        print("🖼️ Ouverture de la fenêtre graphique...")
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"❌ Impossible de charger le cube. Erreur système : {e}")

if __name__ == "__main__":
    interactive_predicted_cube_viewer()