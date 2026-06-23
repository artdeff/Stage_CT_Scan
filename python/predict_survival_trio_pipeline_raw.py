"""
=============================================================================
MODULE : predict_survival_trio_pipeline.py
RÔLE   : Charger DenseNet, EfficientNet et ResNet pour effectuer un vote 
         majoritaire sur le pronostic, ET comparer avec la vraie survie.
=============================================================================
"""

import os
from pathlib import Path
import torch
from collections import Counter
import pandas as pd
import numpy as np
import unicodedata
import re

# Importation des transformations médicales MONAI
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized, ToTensord

# Importation des 3 architectures
from survival_model import create_survival_classifier  # DenseNet
from alternative_survival_model import create_alternative_survival_classifier  # EfficientNet
from third_survival_model import create_resnet_survival_classifier  # ResNet

def get_real_clinical_labels(cube_files):
    """Récupère les vrais labels depuis l'Excel pour vérifier si l'IA a juste."""
    excel_path = r"..\data\PROGNOSTIC RADIOMICS DATABASE.xlsx"
    df_clinique = pd.read_excel(excel_path)

    def normaliser_nom(nom):
        if pd.isna(nom): return ""
        nom_str = str(nom).upper()
        nom_sans_accents = ''.join(c for c in unicodedata.normalize('NFD', nom_str) if unicodedata.category(c) != 'Mn')
        nom_lettres = re.sub(r'[^A-Z\s]', ' ', nom_sans_accents)
        nom_lettres = nom_lettres.replace("SEGMENTARE", "").replace("SEGEMENTARE", "")
        return " ".join(sorted(nom_lettres.split()))

    df_clinique['Nom_Nettoye'] = df_clinique['Nume'].apply(normaliser_nom)
    
    colonne_temps = "PERIOADA SUPRAVIETUIRE (ZILE)"
    mediane_survie = df_clinique[colonne_temps].median()
    df_clinique['Label_Survie'] = np.where(df_clinique[colonne_temps] >= mediane_survie, 1, 0)
    dict_clinique = dict(zip(df_clinique['Nom_Nettoye'], df_clinique['Label_Survie']))
    
    labels_finaux = {}
    for f in cube_files:
        nom_fichier = f.name.replace("cube_", "").replace("_0000.nrrd", "").replace(".nrrd", "").replace("_", " ")
        nom_nettoye = normaliser_nom(nom_fichier)
        if nom_nettoye in dict_clinique:
            labels_finaux[f.name] = int(dict_clinique[nom_nettoye])
        else:
            labels_finaux[f.name] = 0 # Par défaut si non trouvé
    return labels_finaux

def run_trio_survival_prediction():
    print("⏳ Chargement du comité des trois IA pour le vote majoritaire...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    WEIGHTS_DIR = Path(r"..\models\weights")
    
    # 🌟 CORRECTION DU DOSSIER
    CUBES_DIR = Path(r"..\data\05_predicted_cubes")
    
    cube_files = sorted(list(CUBES_DIR.glob("*.nrrd")))
    print(f"📦 Nombre de cubes détectés : {len(cube_files)}\n")
    
    if len(cube_files) == 0:
        print(f"⚠️ Attention : Aucun fichier trouvé dans {CUBES_DIR}")
        return

    # Chargement de la "Correction" (Vraies étiquettes)
    vrais_labels_dict = get_real_clinical_labels(cube_files)

    # 2, 3, 4. Chargement des modèles
    model_dense = create_survival_classifier(spatial_dims=3, in_channels=1, num_classes=2).to(device)
    model_dense.load_state_dict(torch.load(WEIGHTS_DIR / "best_densenet_survival.pt", map_location=device, weights_only=True))
    model_dense.eval()
    
    model_effi = create_alternative_survival_classifier(spatial_dims=3, in_channels=1, num_classes=2).to(device)
    model_effi.load_state_dict(torch.load(WEIGHTS_DIR / "best_efficientnet_survival.pt", map_location=device, weights_only=True))
    model_effi.eval()
    
    model_res = create_resnet_survival_classifier(spatial_dims=3, n_input_channels=1, num_classes=2).to(device)
    model_res.load_state_dict(torch.load(WEIGHTS_DIR / "best_resnet_survival.pt", map_location=device, weights_only=True))
    model_res.eval()
    
    print("✅ Les 3 modèles sont chargés et prêts !\n")
    
    pred_transforms = Compose([
        LoadImaged(keys=["image"], reader="ITKReader"),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(64, 64, 64), mode="bilinear"),
        ToTensord(keys=["image"])
    ])
    
    # En-tête du tableau d'affichage (élargi pour la correction)
    print(f"{'Patient':<25} | {'Dense':<6} | {'Effi':<6} | {'ResNet':<6} || {'Vote IA':<8} | {'Vrai Excel':<10} | {'Résultat':<15}")
    print("-" * 105)
    
    mapping = {0: "Courte", 1: "Longue"}
    bonnes_reponses = 0
    
    with torch.no_grad():
        for cube_path in cube_files:
            # 🌟 CORRECTION DU NOM POUR L'AFFICHAGE (ex: dragomir nicolae)
            patient_name = cube_path.name.replace("cube_", "").replace("_0000.nrrd", "").replace(".nrrd", "").replace("_", " ").title()
            patient_name = patient_name.replace("Segmentare", "").replace("Segementare", "").strip()
            
            vraie_survie_num = vrais_labels_dict[cube_path.name]
            txt_vrai = mapping[vraie_survie_num]
            
            try:
                data = pred_transforms({"image": str(cube_path)})
                input_tensor = data["image"].unsqueeze(0).to(device)
                
                with torch.amp.autocast('cuda'):
                    pred_dense = torch.argmax(model_dense(input_tensor), dim=1).item()
                    pred_effi = torch.argmax(model_effi(input_tensor), dim=1).item()
                    pred_res = torch.argmax(model_res(input_tensor), dim=1).item()
                
                # Calcul du vote majoritaire
                votes = [pred_dense, pred_effi, pred_res]
                vote_majoritaire = Counter(votes).most_common(1)[0][0]
                
                txt_dense = "C" if pred_dense == 0 else "L"
                txt_effi = "C" if pred_effi == 0 else "L"
                txt_res = "C" if pred_res == 0 else "L"
                txt_final = mapping[vote_majoritaire]
                
                # Vérification : Le vote majoritaire a-t-il trouvé la vraie réponse ?
                if vote_majoritaire == vraie_survie_num:
                    statut = "✅ SUCCÈS"
                    bonnes_reponses += 1
                else:
                    statut = "❌ ÉCHEC"
                    
                print(f"{patient_name[:25]:<25} | {txt_dense:<6} | {txt_effi:<6} | {txt_res:<6} || {txt_final:<8} | {txt_vrai:<10} | {statut}")
                
            except Exception as e:
                print(f"{patient_name[:25]:<25} | ❌ Erreur : {e}")  

    print("-" * 105)
    accuracy = (bonnes_reponses / len(cube_files)) * 100
    print(f"🏆 Score final du Comité IA : {bonnes_reponses}/{len(cube_files)} correctes ({accuracy:.1f}%)")

if __name__ == "__main__":
    run_trio_survival_prediction()