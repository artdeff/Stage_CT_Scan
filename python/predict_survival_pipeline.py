"""
=============================================================================
MODULE : predict_survival_pipeline.py
RÔLE   : Prendre les 56 cubes d'IA purifiés -> Appliquer un redimensionnement 
         de sécurité -> Inférence DenseNet3D de survie sans plantage.
=============================================================================
"""

import torch
import torch.nn as nn
from pathlib import Path
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized, ToTensord
from survival_model import create_survival_classifier

# 1. Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUBES_DIR = Path(r"..\data\05_predicted_cubes")
MODEL_PATH = Path(r"..\models\weights\best_densenet_survival.pt")

def run_survival_prediction():
    print("⏳ Initialisation du pipeline de prédiction de survie robuste...")
    
    # 2. Chargement du modèle de classification DenseNet3D
    model = create_survival_classifier(spatial_dims=3, in_channels=1, num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("✅ Modèle DenseNet3D de survie chargé !")

    # 3. Pipeline d'inférence mis à jour avec un REDIMENSIONNEMENT (Resize)
    # On force chaque cube à adopter une taille standard (ex: 64x64x64) pour que le DenseNet
    # puisse faire ses convolutions et ses pooling sans jamais manquer de pixels.
    pred_transforms = Compose([
        LoadImaged(keys=["image"], reader="ITKReader"),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(64, 64, 64), mode="trilinear"), # <-- LA SÉCURITÉ EST ICI
        ToTensord(keys=["image"])
    ])

    cube_files = sorted(list(CUBES_DIR.glob("*.nrrd")))
    print(f"📦 Nombre de cubes à analyser : {len(cube_files)}\n")
    print(f"{'Patient ID':<25} | {'Prédiction IA (Pronostic)':<30}")
    print("-" * 60)

    for cube_path in cube_files:
        patient_id = cube_path.stem.replace("cube_raw_", "")
        
        try:
            # Application des transformations (le dictionnaire d'entrée)
            data = pred_transforms({"image": str(cube_path)})
            input_tensor = data["image"].unsqueeze(0).to(device)
            
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    outputs = model(input_tensor)
                    prediction = torch.argmax(outputs, dim=1).item()
            
            pronostic = "Survie Longue (>= Seuil)" if prediction == 1 else "Survie Courte (< Seuil)"
            print(f"{patient_id:<25} | {pronostic:<30}")
            
        except Exception as e:
            print(f"{patient_id:<25} | ❌ Erreur d'analyse : {e}")

if __name__ == "__main__":
    run_survival_prediction()