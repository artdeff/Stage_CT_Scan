"""
=============================================================================
MODULE : view_extracted_cube.py
RÔLE   : Menu interactif pour charger un cube de pancréas extrait (.nrrd)
         et afficher sa coupe centrale pour contrôle qualité.
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from monai.transforms import LoadImage

CUBES_DIR = Path(r"..\data\03_processed\pancreas_cubes")

def interactive_cube_viewer():
    print("\n" + "="*50)
    print("🔍 PANCREAS CUBES DISPONIBLES POUR CONTRÔLE :")
    print("="*50)
    
    cubes = sorted(list(CUBES_DIR.glob("*.nrrd")))
    if not cubes:
        print("❌ Aucun cube trouvé dans le dossier pancreas_cubes !")
        return
        
    for i, c in enumerate(cubes):
        print(f"  [{i}] -> {c.name}")
    print("="*50)
    
    # Choix du cube
    while True:
        try:
            choix = input("\n👉 Choisis le numéro du cube à visualiser (ex: 0) : ")
            idx = int(choix)
            if 0 <= idx < len(cubes):
                selected_cube = cubes[idx]
                break
            print(f"❌ Numéro invalide. Entre un chiffre entre 0 et {len(cubes)-1}.")
        except ValueError:
            print("❌ Entre un nombre entier.")
            
    print(f"\n⏳ Chargement du volume du cube : {selected_cube.name}...")
    
    # Chargement
    loader = LoadImage(reader="ITKReader", image_only=True)
    cube_array = loader(str(selected_cube)).numpy()
    
    print(f"📐 Dimensions de ce cube de pancréas (X, Y, Z) : {cube_array.shape}")
    
    # Extraction de la coupe du milieu sur l'axe Z du cube
    mid_z = cube_array.shape[2] // 2
    img_slice = cube_array[:, :, mid_z]
    
    # Affichage graphique
    plt.figure(figsize=(6, 6))
    # Nos valeurs sont déjà normalisées entre 0.0 et 1.0 par la pipeline
    plt.imshow(img_slice.T, cmap="hot") 
    plt.colorbar(label="Intensité normalisée")
    plt.title(f"Coupe centrale Z={mid_z} du Pancréas Isolé\nFichier : {selected_cube.name}")
    plt.axis("off")
    
    print("🖼️ Ouverture de la fenêtre graphique...")
    plt.show()

if __name__ == "__main__":
    interactive_cube_viewer()