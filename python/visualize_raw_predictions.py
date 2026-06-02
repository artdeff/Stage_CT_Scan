"""
=============================================================================
MODULE : visualize_raw_predictions.py (Version Sécurisée SimpleITK)
RÔLE   : Menu interactif pour inspecter visuellement les prédictions (masques) 
         de l'IA U-Net superposées sur les scanners bruts transformés,
         sans utiliser les modules ITK de MONAI bloqués par le système.
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import SimpleITK as sitk  # <-- Sécurité anti-blocage applicatif Windows

# 1. Configuration des chemins
INTERIM_DIR = Path(r"..\data\02_interim")      
PREDICTIONS_DIR = Path(r"..\data\04_output")   

def resample_image_to_match_mask(sitk_image, target_spacing=(1.5, 1.5, 2.0)):
    """
    Rééchantillonne le scanner brut à la volée avec SimpleITK pour reproduire 
    EXACTEMENT l'espace géométrique (Spacing et RAS) utilisé par l'IA.
    """
    # 1. Forcer l'orientation RAS (comme le faisait Orientationd de MONAI)
    sitk_image = sitk.DICOMOrient(sitk_image, "RAS")
    
    # 2. Calculer les nouvelles dimensions pour correspondre au spacing 1.5x1.5x2.0
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()
    
    new_size = [
        int(round(original_size[0] * original_spacing[0] / target_spacing[0])),
        int(round(original_size[1] * original_spacing[1] / target_spacing[1])),
        int(round(original_size[2] * original_spacing[2] / target_spacing[2]))
    ]
    
    # 3. Interpolation trilinéaire
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    
    return resample.Execute(sitk_image)

def interactive_prediction_viewer():
    print("\n" + "="*60)
    print("👁️ VISUALISATION DES PRÉDICTIONS (Version Sécurisée SimpleITK) :")
    print("="*60)
    
    pred_files = sorted(list(PREDICTIONS_DIR.glob("pred_*.nrrd")))
    if not pred_files:
        print("❌ Aucun masque trouvé dans 04_output.")
        return
        
    for i, f in enumerate(pred_files):
        patient_id = f.stem.replace("pred_", "")
        print(f"  [{i}] -> Patient : {patient_id}")
    print("="*60)
    
    while True:
        try:
            choix = input("\n👉 Choisis le numéro du patient à inspecter (ex: 0) : ")
            idx = int(choix)
            if 0 <= idx < len(pred_files):
                selected_pred = pred_files[idx]
                patient_id = selected_pred.stem.replace("pred_", "")
                selected_image = INTERIM_DIR / f"{patient_id}.nrrd"
                break
            print(f"❌ Numéro invalide. Choisis entre 0 et {len(pred_files)-1}.")
        except ValueError:
            print("❌ Entre un nombre entier.")

    print(f"\n⏳ Chargement et alignement géométrique pour {patient_id}...")

    try:
        # 1. Chargement de l'image brute et rééchantillonnage de sécurité
        sitk_img = sitk.ReadImage(str(selected_image))
        sitk_img_resampled = resample_image_to_match_mask(sitk_img, target_spacing=(1.5, 1.5, 2.0))
        
        # 2. Chargement du masque IA
        sitk_mask = sitk.ReadImage(str(selected_pred))
        
        # 3. Conversion en matrices numpy homogènes (ITK Z,Y,X -> Numpy X,Y,Z)
        img_vol = sitk.GetArrayFromImage(sitk_img_resampled).transpose(2, 1, 0)
        mask_vol = sitk.GetArrayFromImage(sitk_mask).transpose(2, 1, 0)
        
        print(f"📐 Dimensions alignées de l'image : {img_vol.shape}")
        print(f"📐 Dimensions alignées du masque : {mask_vol.shape}")
        
        # 4. Détection de la coupe centrale du pancréas
        z_indices = np.where(mask_vol > 0)[2] 
        
        if len(z_indices) > 0:
            mid_z = int(np.median(z_indices))
            print(f"🎯 Pancréas détecté ! Centrage automatique sur la coupe Z={mid_z}")
        else:
            mid_z = img_vol.shape[2] // 2
            print(f"⚠️ L'IA n'a rien détecté. Affichage du centre par défaut Z={mid_z}")

        # 5. Extraction de la coupe 2D
        img_slice = img_vol[:, :, mid_z]
        mask_slice = mask_vol[:, :, mid_z]
        
        # Masquage transparent du fond
        mask_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)

        # 6. Affichage graphique Matplotlib
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle(f"Contrôle Qualité IA - Patient {patient_id} | Coupe Z={mid_z}", fontsize=14, fontweight='bold')
        
        VMIN, VMAX = -150, 250  # Fenêtrage Hounsfield pancréas

        axes[0].imshow(img_slice.T, cmap='gray', vmin=VMIN, vmax=VMAX)
        axes[0].set_title("Scanner Original Re-spacé")
        axes[0].axis('off')

        axes[1].imshow(img_slice.T, cmap='gray', vmin=VMIN, vmax=VMAX)
        axes[1].imshow(mask_overlay.T, cmap='autumn', alpha=0.5, interpolation='none')
        axes[1].set_title("Superposition Masque IA (U-Net)")
        axes[1].axis('off')

        plt.tight_layout()
        print("🖼️ Ouverture de la fenêtre graphique...")
        plt.show()
        
    except Exception as e:
        print(f"❌ Erreur lors de l'alignement ou de l'affichage : {e}")

if __name__ == "__main__":
    interactive_prediction_viewer()