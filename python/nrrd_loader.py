"""
=============================================================================
MODULE : nrrd_loader.py
RÔLE   : Charger les volumes NRRD et leurs masques, et vérifier l'alignement.
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from monai.transforms import LoadImage
from monai.data import ITKReader  # 🌟 NOUVEAU : On importe le lecteur spécifique

def load_and_verify_nrrd(image_path: str | Path, mask_path: str | Path):
    """
    Charge une image CT et son masque de segmentation, et affiche une coupe superposée.
    """
    print(f"⏳ Chargement de l'image : {image_path.name}")
    print(f"⏳ Chargement du masque : {mask_path.name}")
    
    # Initialisation du chargeur MONAI (image_only=False pour récupérer les métadonnées)
    loader = LoadImage(image_only=False)
    
    # Chargement des données
    image_vol, image_meta = loader(str(image_path))
    mask_vol, mask_meta = loader(str(mask_path))
    
    # Conversion en Numpy pour l'affichage (MONAI renvoie parfois des MetaTensors)
    image_vol = np.array(image_vol)
    mask_vol = np.array(mask_vol)

    labels_presents = np.unique(mask_vol)
    print(f"🏷️ Classes (Labels) trouvées dans le masque : {labels_presents}")
    
    print("\n✅ Chargement terminé !")
    print(f"📏 Dimensions de l'image (X, Y, Z) : {image_vol.shape}")
    print(f"📏 Dimensions du masque (X, Y, Z) : {mask_vol.shape}")
    
    # ⚠️ Vérification critique : les dimensions doivent être rigoureusement identiques
    if image_vol.shape != mask_vol.shape:
        print("🚨 ALERTE : L'image et le masque n'ont pas la même taille !")
        return
        
    # --- Recherche intelligente de la meilleure coupe ---
    # On calcule la somme des pixels du masque pour chaque coupe Z
    pixels_par_coupe = np.sum(mask_vol, axis=(0, 1))
    best_z = np.argmax(pixels_par_coupe) # L'indice Z avec le plus de pixels
    
    if pixels_par_coupe[best_z] == 0:
        print("⚠️ ALERTE : Le masque est complètement vide (aucun pixel colorié) !")
        mid_z = image_vol.shape[2] // 2
    else:
        mid_z = best_z
        print(f"🎯 Organe/Tumeur détecté ! Affichage de la coupe optimale : Z = {mid_z}")

    img_slice = image_vol[:, :, mid_z]
    mask_slice = mask_vol[:, :, mid_z]
    
    plt.figure(figsize=(12, 6))
    
    # 1. Affichage Image seule
    plt.subplot(1, 2, 1)
    plt.imshow(img_slice.T, cmap='gray', vmin=-150, vmax=250) # Fenêtrage Tissus Mous
    plt.title(f"Image CT brute (Coupe Z={mid_z})")
    plt.axis('off')
    
    # 2. Affichage Superposition (Image + Masque)
    plt.subplot(1, 2, 2)
    plt.imshow(img_slice.T, cmap='gray', vmin=-150, vmax=250)
    # On superpose le masque en rouge (alpha = transparence)
    # mask_slice > 0 permet d'ignorer le fond noir du masque
    masked_data = np.ma.masked_where(mask_slice.T == 0, mask_slice.T)
    plt.imshow(masked_data, cmap='autumn', alpha=0.5, interpolation='none')
    plt.title("Contrôle Qualité : Image + Segmentation")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# =========================================================================
# BLOC DE TEST
# =========================================================================
if __name__ == "__main__":
    # 1. On pointe DIRECTEMENT vers le dossier du patient qui contient les deux fichiers
    dossier_patient = Path(r"C:\Users\Arthur\Documents\Travail\Polytech\4A\Stage\CT Database\SEGMENTARI VENOS DOC 1\bana marian segmentare")
    
    # 2. On cible précisément les fichiers à l'intérieur de ce dossier
    # ⚠️ Vérifie que le nom de l'image est bien celui-là dans ce dossier précis !
    mon_image = dossier_patient / "80448 TAP AX V.nrrd" 
    mon_masque = dossier_patient / "Segmentation.seg.nrrd"
    
    print(f"📂 Dossier cible : {dossier_patient}")
    
    try:
        load_and_verify_nrrd(mon_image, mon_masque)
    except FileNotFoundError as e:
        print(f"❌ Fichier introuvable. Vérifie tes chemins Windows :\n{e}")