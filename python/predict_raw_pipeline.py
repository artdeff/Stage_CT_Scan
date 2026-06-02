"""
=============================================================================
MODULE : predict_raw_pipeline.py
RÔLE   : Pipeline d'inférence en masse sur les données brutes converties. 
         Prend le .nrrd brut -> IA U-Net 3D -> Sauvegarde le masque .nrrd.
=============================================================================
"""

import os
from pathlib import Path
import torch
import numpy as np
import SimpleITK as sitk
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, ScaleIntensityRanged
from monai.inferers import sliding_window_inference
from unet_segmentation import create_unet_model

# 1. Configuration des chemins et périphériques
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INTERIM_DIR = Path(r"..\data\02_interim")
OUTPUT_DIR = Path(r"..\data\04_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = Path(r"..\models\weights\best_unet_model.pt")

def run_inference_pipeline():
    print(f"🚀 Début de l'inférence de masse sur la base brute...")
    print(f"🖥️ Périphérique : {device}")
    print(f"📂 Source : {INTERIM_DIR.resolve()}")
    print(f"📂 Destination : {OUTPUT_DIR.resolve()}\n")

    # 2. Chargement du modèle U-Net 3D
    print("⏳ Chargement du cerveau de l'IA (Poids du modèle)...")
    model = create_unet_model(spatial_dims=3, in_channels=1, out_channels=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("✅ Modèle chargé avec succès !\n")

    # 3. Pipeline de transformation
    # On applique EXACTEMENT les mêmes transformations géométriques que pendant l'entraînement
    inference_transforms = Compose([
        LoadImaged(keys=["image"], reader="ITKReader"),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear")),
        ScaleIntensityRanged(keys=["image"], a_min=-150, a_max=250, b_min=0.0, b_max=1.0, clip=True)
    ])

    # 4. Parcours des fichiers patients bruts
    patient_files = sorted(list(INTERIM_DIR.glob("*.nrrd")))
    print(f"📦 Nombre de patients à segmenter : {len(patient_files)}")

    for i, file_path in enumerate(patient_files):
        patient_id = file_path.stem  # ex: raw_1440212513493
        output_filename = OUTPUT_DIR / f"pred_{patient_id}.nrrd"

        # On saute si déjà fait (pratique si ça plante au milieu et qu'on doit relancer)
        if output_filename.exists():
            print(f"  ⏩ [{i+1}/{len(patient_files)}] Le patient {patient_id} a déjà été traité. On passe.")
            continue

        print(f"  🔍 [{i+1}/{len(patient_files)}] Analyse de {patient_id}...")
        
        try:
            # Application des transformations
            data_dict = {"image": str(file_path)}
            transformed_data = inference_transforms(data_dict)
            
            # (C, H, W, D) -> (1, C, H, W, D) pour créer un "batch" de 1 patient pour PyTorch
            input_tensor = transformed_data["image"].unsqueeze(0).to(device)

            # Inférence par fenêtre glissante (pour ne pas saturer les 8 Go de VRAM de la RTX 5070)
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    outputs = sliding_window_inference(input_tensor, roi_size=(96, 96, 96), sw_batch_size=4, predictor=model)
                    
                    # On récupère la classe ayant la plus forte probabilité (0, 1, 2, 3)
                    pred_mask = torch.argmax(outputs, dim=1).detach().cpu().numpy()[0] # Résultat: (H, W, D)

            # 5. Sauvegarde au format .nrrd avec SimpleITK
            # Le tenseur pred_mask est en format (X, Y, Z) (Orientation RAS de MONAI). 
            # SimpleITK attend des arrays numpy en (Z, Y, X), on transpose donc la matrice.
            pred_mask_sitk = pred_mask.transpose(2, 1, 0).astype(np.uint8)
            sitk_image = sitk.GetImageFromArray(pred_mask_sitk)
            
            # Enregistrement du spacing physique (que nous avons forcé à 1.5, 1.5, 2.0 plus haut)
            sitk_image.SetSpacing((1.5, 1.5, 2.0))
            
            # Écriture physique du fichier
            sitk.WriteImage(sitk_image, str(output_filename))
            print(f"    ✅ Export réussi -> {output_filename.name}")
            
        except Exception as e:
            print(f"    ❌ Échec pour {patient_id}. Erreur : {e}")

    print("\n🎉 Mission accomplie ! Toute la base de données brute a été segmentée par ton IA.")

if __name__ == "__main__":
    run_inference_pipeline()