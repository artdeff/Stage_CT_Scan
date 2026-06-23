"""
=============================================================================
MODULE : predict_survival_dual_pipeline.py
RÔLE   : Charger DenseNet3D ET EfficientNet3D -> Comparer leurs prédictions 
         de survie sur les 56 patients bruts pour double-validation.
=============================================================================
"""

import torch
from pathlib import Path
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized, ToTensord

# Import des deux modèles créés
from survival_model import create_survival_classifier               # Modèle 1
from alternative_survival_model import create_alternative_survival_classifier # Modèle 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUBES_DIR = Path(r"..\data\05_predicted_cubes")

# Chemins des deux fichiers de poids (Une fois entraînés sur le vrai Excel)
MODEL_DENSE_PATH = Path(r"..\models\weights\best_densenet_survival.pt")
MODEL_EFFI_PATH = Path(r"..\models\weights\best_efficientnet_survival.pt")

def run_dual_survival_prediction():
    print("⏳ Chargement des deux intelligences artificielles pour validation croisée...")
    
    # 1. Init et chargement DenseNet3D
    model_dense = create_survival_classifier(spatial_dims=3, in_channels=1, num_classes=2)
    if MODEL_DENSE_PATH.exists():
        model_dense.load_state_dict(torch.load(MODEL_DENSE_PATH, map_location=device))
    model_dense.to(device).eval()
    
    # 2. Init et chargement EfficientNet3D
    model_effi = create_alternative_survival_classifier(spatial_dims=3, in_channels=1, num_classes=2)
    if MODEL_EFFI_PATH.exists():
        model_effi.load_state_dict(torch.load(MODEL_EFFI_PATH, map_location=device))
    model_effi.to(device).eval()
    
    print("✅ Modèles DenseNet3D et EfficientNet3D chargés et prêts !")

    # 3. Pipeline de transformation
    pred_transforms = Compose([
        LoadImaged(keys=["image"], reader="ITKReader"),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(64, 64, 64), mode="trilinear"),
        ToTensord(keys=["image"])
    ])

    cube_files = sorted(list(CUBES_DIR.glob("*.nrrd")))
    print(f"📦 Nombre de cubes à analyser : {len(cube_files)}\n")
    
    # En-tête du tableau de comparaison
    print(f"{'Patient ID':<20} | {'DenseNet3D':<20} | {'EfficientNet3D':<20} | {'Statut'}")
    print("-" * 80)

    for cube_path in cube_files:
        patient_id = cube_path.stem.replace("cube_raw_", "")
        
        try:
            data = pred_transforms({"image": str(cube_path)})
            input_tensor = data["image"].unsqueeze(0).to(device)
            
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    # Inférence modèle 1
                    out_dense = model_dense(input_tensor)
                    pred_dense = torch.argmax(out_dense, dim=1).item()
                    
                    # Inférence modèle 2
                    out_effi = model_effi(input_tensor)
                    pred_effi = torch.argmax(out_effi, dim=1).item()
            
            txt_dense = "Longue" if pred_dense == 1 else "Courte"
            txt_effi = "Longue" if pred_effi == 1 else "Courte"
            
            # Comparaison de l'accord des modèles
            accord = "🤝 OK (Accord)" if pred_dense == pred_effi else "⚠️ Désaccord !"
            
            print(f"{patient_id:<20} | {txt_dense:<20} | {txt_effi:<20} | {accord}")
            
        except Exception as e:
            print(f"{patient_id:<20} | ❌ Erreur : {e}")

if __name__ == "__main__":
    run_dual_survival_prediction()