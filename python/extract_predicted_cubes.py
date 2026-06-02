"""
=============================================================================
MODULE : extract_predicted_cubes.py
RÔLE   : Extraire la boîte englobante (bounding box) des 56 patients bruts
         en utilisant les masques générés par l'IA (U-Net).
         Intègre un post-traitement SimpleITK pour éliminer le bruit (faux positifs).
=============================================================================
"""

import os
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, ScaleIntensityRanged

# 1. Configuration des chemins
INTERIM_DIR = Path(r"..\data\02_interim")          # Les images brutes converties (.nrrd)
PREDICTIONS_DIR = Path(r"..\data\04_output")       # Les masques prédits par le U-Net
CUBES_OUTPUT_DIR = Path(r"..\data\05_predicted_cubes") # Là où on va ranger les cubes finaux
CUBES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def keep_largest_connected_component(sitk_mask):
    """
    PREMIÈRE AJOUT : Cette fonction élimine les petites taches satellites
    et ne garde que le plus gros bloc 3D (le vrai pancréas).
    """
    # Cast obligatoire en UInt8/Int32 pour SimpleITK
    sitk_mask = sitk.Cast(sitk_mask, sitk.sitkUInt8)
    
    # 1. On sépare le masque en "îles" indépendantes
    connected_components = sitk.ConnectedComponent(sitk_mask)
    
    # 2. On trie les îles par taille (la plus grande aura l'ID 1)
    labeled_components = sitk.RelabelComponent(connected_components)
    
    # 3. On ne garde que l'île ID 1 (le reste devient du fond à 0)
    cleaned_mask = labeled_components == 1
    
    return cleaned_mask


def extract_bounding_box_3d(image_np, label_np, margin=4):
    """Calcule les coordonnées de la boîte englobante autour du masque nettoyé."""
    indices = np.argwhere(label_np > 0)
    
    if indices.size == 0:
        return None # L'IA n'a rien trouvé du tout sur ce patient
        
    # Bornes minimales et maximales
    z_min, y_min, x_min = indices.min(axis=0)
    z_max, y_max, x_max = indices.max(axis=0)
    
    # Ajout de la marge (Padding)
    z_min, y_min, x_min = max(0, z_min - margin), max(0, y_min - margin), max(0, x_min - margin)
    z_max = min(image_np.shape[0], z_max + margin + 1)
    y_max = min(image_np.shape[1], y_max + margin + 1)
    x_max = min(image_np.shape[2], x_max + margin + 1)
    
    # Découpage du cube
    return image_np[z_min:z_max, y_min:y_max, x_min:x_max]


def run_extraction():
    print("🚀 Début de l'extraction des cubes standardisés et purifiés...")
    
    prep_pipeline = Compose([
        LoadImaged(keys=["image"], reader="ITKReader"),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
        # Très important pour le futur DenseNet : intensité normalisée entre 0.0 et 1.0 !
        ScaleIntensityRanged(keys=["image"], a_min=-150, a_max=250, b_min=0.0, b_max=1.0, clip=True) 
    ])

    predicted_masks = sorted(list(PREDICTIONS_DIR.glob("pred_*.nrrd")))

    for i, mask_path in enumerate(predicted_masks):
        patient_id = mask_path.stem.replace("pred_", "")
        image_path = INTERIM_DIR / f"{patient_id}.nrrd"
        output_path = CUBES_OUTPUT_DIR / f"cube_{patient_id}.nrrd"
        
        #if output_path.exists():
        #    print(f"  ⏩ [{i+1}/{len(predicted_masks)}] {patient_id} déjà extrait. On passe.")
        #    continue
            
        if not image_path.exists():
            print(f"  ❌ [{i+1}/{len(predicted_masks)}] Image originale introuvable pour {patient_id}")
            continue
            
        print(f"  ✂️ [{i+1}/{len(predicted_masks)}] Extraction et filtrage de {patient_id}...")
        
        try:
            # 1. Pipeline MONAI sur l'image originale
            data = prep_pipeline({"image": str(image_path)})
            img_np = data["image"][0].numpy()
            
            # 2. Chargement du masque IA brut
            sitk_mask = sitk.ReadImage(str(mask_path))
            
            # DEUXIÈME AJOUT : On applique la purification ici !
            sitk_mask_cleaned = keep_largest_connected_component(sitk_mask)
            
            # Conversion en numpy pour l'extraction (ITK Z,Y,X -> Numpy X,Y,Z)
            mask_np = sitk.GetArrayFromImage(sitk_mask_cleaned).transpose(2, 1, 0)
            
            # 3. Découpage tridimensionnel
            pancreas_cube = extract_bounding_box_3d(img_np, mask_np, margin=4)
            
            if pancreas_cube is not None:
                # Retour au format ITK pour la sauvegarde (Numpy X,Y,Z -> ITK Z,Y,X)
                sitk_out = sitk.GetImageFromArray(pancreas_cube.transpose(2, 1, 0)) 
                sitk.WriteImage(sitk_out, str(output_path))
                print(f"    ✅ Cube sauvegardé sans bruit ! Dimensions : ({pancreas_cube.shape})")
            else:
                print(f"    ⚠️ Aucun pancréas détecté après nettoyage.")
                
        except Exception as e:
            print(f"    ❌ Échec du traitement pour {patient_id}. Erreur : {e}")

    print("\n🎉 Tous tes cubes d'IA sont extraits, nettoyés du bruit et prêts pour la prédiction de survie !")

if __name__ == "__main__":
    run_extraction()