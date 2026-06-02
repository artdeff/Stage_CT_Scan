"""
=============================================================================
MODULE : predict_and_visualize.py (Version Sécurisée SimpleITK)
RÔLE   : Menu interactif pour évaluer le modèle U-Net sur les données de 
         validation. Charge l'image, applique l'IA et compare au masque médecin.
         Immunisé contre les blocages de DLL Windows.
=============================================================================
"""

import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from unet_segmentation import create_unet_model

# 1. Configuration des chemins et périphériques
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path(r"..\data\03_processed")
IMAGES_DIR = DATA_DIR / "imagesTr"
LABELS_DIR = DATA_DIR / "labelsTr"
MODEL_PATH = Path(r"..\models\weights\best_unet_model.pt")

def resample_and_normalize_sitk(sitk_image, target_spacing=(1.5, 1.5, 2.0)):
    """Rééchantillonne et normalise l'intensité avec SimpleITK (Simule le pipeline IA)."""
    # Orientation RAS
    sitk_image = sitk.DICOMOrient(sitk_image, "RAS")
    
    # Calcul des nouvelles dimensions pour le spacing target
    orig_spacing = sitk_image.GetSpacing()
    orig_size = sitk_image.GetSize()
    new_size = [
        int(round(orig_size[0] * orig_spacing[0] / target_spacing[0])),
        int(round(orig_size[1] * orig_spacing[1] / target_spacing[1])),
        int(round(orig_size[2] * orig_spacing[2] / target_spacing[2]))
    ]
    
    # Resample Image
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    img_resampled = resample.Execute(sitk_image)
    
    # Conversion Numpy + Normalisation d'intensité (-150 à 250 HU -> 0.0 à 1.0)
    img_np = sitk.GetArrayFromImage(img_resampled).transpose(2, 1, 0)
    img_np = np.clip(img_np, -150, 250)
    img_np = (img_np - (-150)) / (250 - (-150))
    
    return img_np, img_resampled

def resample_label_sitk(sitk_label, reference_sitk_image):
    """Rééchantillonne le masque du médecin sur la même grille que l'image traitée."""
    sitk_label = sitk.DICOMOrient(sitk_label, "RAS")
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkNearestNeighbor) # Proche voisin pour les labels discrets
    resample.SetReferenceImage(reference_sitk_image)
    label_resampled = resample.Execute(sitk_label)
    return sitk.GetArrayFromImage(label_resampled).transpose(2, 1, 0)


def evaluate_and_plot():
    print("\n🔮 Initialisation de l'inférence sécurisée sur :", device)
    
    # 2. Liste des patients de validation
    if not IMAGES_DIR.exists():
        print(f"❌ Dossier introuvable : {IMAGES_DIR.resolve()}")
        return
        
    images_val = sorted(list(IMAGES_DIR.glob("*.nrrd")))
    print("\n" + "="*50)
    print("📂 PATIENTS DE VALIDATION DISPONIBLES :")
    print("="*50)
    for i, f in enumerate(images_val):
        print(f"  [{i}] -> {f.name}")
    print("="*50)
    
    while True:
        try:
            choix = input("\n👉 Tape le numéro du patient que tu souhaites analyser (ex: 0) : ")
            idx = int(choix)
            if 0 <= idx < len(images_val):
                img_path = images_val[idx]
                
                # --- LE CORRECTIF NOM DE FICHIER EST ICI ---
                # On remplace "patient_XXX_0000.nrrd" par "patient_XXX.nrrd" pour le dossier médecin
                lbl_name = img_path.name.replace("_0000", "")
                lbl_path = LABELS_DIR / lbl_name
                # --------------------------------------------
                break
            print(f"❌ Index invalide.")
        except ValueError:
            print("❌ Entre un entier.")

    print(f"\n⏳ Chargement et traitement sécurisé de {img_path.name}...")
    
    try:
        # 3. Chargement et normalisation géométrique via SimpleITK
        sitk_img = sitk.ReadImage(str(img_path))
        img_np, sitk_img_ref = resample_and_normalize_sitk(sitk_img)
        
        # Masque médecin
        has_label = False
        if lbl_path.exists():
            sitk_lbl = sitk.ReadImage(str(lbl_path))
            lbl_np = resample_label_sitk(sitk_lbl, sitk_img_ref)
            if np.max(lbl_np) > 0:
                has_label = True
        else:
            lbl_np = np.zeros_like(img_np)
            print(f"⚠️ Note : Masque du médecin d'origine introuvable ({lbl_path.name}).")

        # 4. Chargement de l'IA (U-Net) et inférence
        print("⏳ Chargement du modèle U-Net...")
        model = create_unet_model(spatial_dims=3, in_channels=1, out_channels=4)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        
        # Préparation du tenseur PyTorch (1, C, X, Y, Z)
        input_tensor = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0).to(device)
        
        print("🧠 Inférence de l'IA en cours...")
        from monai.inferers import sliding_window_inference
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                outputs = sliding_window_inference(input_tensor, roi_size=(96, 96, 96), sw_batch_size=4, predictor=model)
                pred_np = torch.argmax(outputs, dim=1).detach().cpu().numpy()[0]

        # 5. Calcul de la coupe Z
        if has_label:
            z_indices = np.where(lbl_np > 0)[2]
            mid_z = int(np.median(z_indices))
            print(f"🎯 Masque médecin trouvé ! Centrage sur la coupe : Z={mid_z}")
        else:
            z_indices = np.where(pred_np > 0)[2]
            mid_z = int(np.median(z_indices)) if len(z_indices) > 0 else img_np.shape[2] // 2
            print(f"🎯 Centrage par défaut / IA : Z={mid_z}")

        # Extraction des coupes 2D
        slice_img = img_np[:, :, mid_z]
        slice_lbl = lbl_np[:, :, mid_z]
        slice_pred = pred_np[:, :, mid_z]

        # Masquage transparent pour l'affichage
        overlay_lbl = np.ma.masked_where(slice_lbl == 0, slice_lbl)
        overlay_pred = np.ma.masked_where(slice_pred == 0, slice_pred)

        # 6. Affichage graphique triple
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Évaluation U-Net - {img_path.name} | Coupe Z={mid_z}", fontsize=14, fontweight='bold')

        # Scanner
        axes[0].imshow(slice_img.T, cmap='gray', vmin=0.0, vmax=1.0)
        axes[0].set_title("Scanner Normalisé")
        axes[0].axis('off')

        # Vérité Terrain (Médecin)
        axes[1].imshow(slice_img.T, cmap='gray', vmin=0.0, vmax=1.0)
        if has_label and np.sum(slice_lbl) > 0:
            axes[1].imshow(overlay_lbl.T, cmap='winter', alpha=0.6, interpolation='none', vmin=1, vmax=4)
            axes[1].set_title("Masque original du Médecin")
        else:
            axes[1].set_title("Masque Médecin (Absent sur cette coupe)")
        axes[1].axis('off')

        # Prédiction IA
        axes[2].imshow(slice_img.T, cmap='gray', vmin=0.0, vmax=1.0)
        axes[2].imshow(overlay_pred.T, cmap='autumn', alpha=0.5, interpolation='none', vmin=1, vmax=4)
        axes[2].set_title("Segmentation calculée par l'IA")
        axes[2].axis('off')

        plt.tight_layout()
        print("🖼️ Affichage de la fenêtre d'analyse...")
        plt.show()

    except Exception as e:
        print(f"❌ Échec de l'évaluation : {e}")

if __name__ == "__main__":
    evaluate_and_plot()