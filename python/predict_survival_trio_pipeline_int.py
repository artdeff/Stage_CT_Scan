"""
=============================================================================
MODULE : predict_survival_trio_pipeline.py
RÔLE   : Charger DenseNet, EfficientNet et ResNet pour effectuer un vote 
         majoritaire sur le pronostic (3 Classes), ET comparer avec l'Excel.
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

from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized, ToTensord

from survival_model import create_survival_classifier
from alternative_survival_model import create_alternative_survival_classifier
from third_survival_model import create_resnet_survival_classifier

def get_real_clinical_labels(cube_files):
    """Récupère les vrais labels (3 classes) depuis l'Excel pour la correction."""
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
    
    # --- 1. RÈGLE DES 3 CLASSES (Tertiles) ---
    seuil_bas = df_clinique[colonne_temps].quantile(0.33)
    seuil_haut = df_clinique[colonne_temps].quantile(0.67)
    
    conditions = [
        df_clinique[colonne_temps] < seuil_bas,
        (df_clinique[colonne_temps] >= seuil_bas) & (df_clinique[colonne_temps] <= seuil_haut),
        df_clinique[colonne_temps] > seuil_haut
    ]
    df_clinique['Label_Survie'] = np.select(conditions, [0, 1, 2])
    # ------------------------------------------
    
    dict_clinique = dict(zip(df_clinique['Nom_Nettoye'], df_clinique['Label_Survie']))
    
    labels_finaux = {}
    for f in cube_files:
        nom_fichier = f.name.replace("cube_", "").replace("_0000.nrrd", "").replace(".nrrd", "").replace("_", " ")
        nom_nettoye = normaliser_nom(nom_fichier)
        if nom_nettoye in dict_clinique:
            labels_finaux[f.name] = int(dict_clinique[nom_nettoye])
        else:
            labels_finaux[f.name] = 0 
    return labels_finaux

def run_trio_survival_prediction():
    print("⏳ Chargement du comité des trois IA pour le vote majoritaire (3 Classes)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    WEIGHTS_DIR = Path(r"..\models\weights")
    CUBES_DIR = Path(r"..\data\03_processed\pancreas_cubes")
    
    cube_files = sorted(list(CUBES_DIR.glob("*.nrrd")))
    print(f"📦 Nombre de cubes détectés : {len(cube_files)}\n")
    
    if len(cube_files) == 0:
        print(f"⚠️ Attention : Aucun fichier trouvé dans {CUBES_DIR}")
        return

    vrais_labels_dict = get_real_clinical_labels(cube_files)

    # --- 2. CHARGEMENT AVEC num_classes=3 ---
    model_dense = create_survival_classifier(spatial_dims=3, in_channels=1, num_classes=3).to(device)
    model_dense.load_state_dict(torch.load(WEIGHTS_DIR / "best_densenet_survival.pt", map_location=device, weights_only=True))
    model_dense.eval()
    
    model_effi = create_alternative_survival_classifier(spatial_dims=3, in_channels=1, num_classes=3).to(device)
    model_effi.load_state_dict(torch.load(WEIGHTS_DIR / "best_efficientnet_survival.pt", map_location=device, weights_only=True))
    model_effi.eval()
    
    model_res = create_resnet_survival_classifier(spatial_dims=3, n_input_channels=1, num_classes=3).to(device)
    model_res.load_state_dict(torch.load(WEIGHTS_DIR / "best_resnet_survival.pt", map_location=device, weights_only=True))
    model_res.eval()
    # ----------------------------------------
    
    print("✅ Les 3 modèles sont chargés et prêts !\n")
    
    pred_transforms = Compose([
        LoadImaged(keys=["image"], reader="ITKReader"),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(64, 64, 64), mode="bilinear"),
        ToTensord(keys=["image"])
    ])
    
    print(f"{'Patient':<25} | {'Dense':<6} | {'Effi':<6} | {'ResNet':<6} || {'Vote IA':<8} | {'Vrai Excel':<10} | {'Résultat':<15}")
    print("-" * 105)
    
    # --- 3. MISE À JOUR DE L'AFFICHAGE (Ajout de Moyenne) ---
    mapping = {0: "Courte", 1: "Moyenne", 2: "Longue"}
    bonnes_reponses = 0
    
    with torch.no_grad():
        for cube_path in cube_files:
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
                
                votes = [pred_dense, pred_effi, pred_res]
                vote_majoritaire = Counter(votes).most_common(1)[0][0]
                
                # Astuce : On prend juste la première lettre ("C", "M", ou "L") pour les petites colonnes
                txt_dense = mapping[pred_dense][0]
                txt_effi = mapping[pred_effi][0]
                txt_res = mapping[pred_res][0]
                
                txt_final = mapping[vote_majoritaire]
                
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
    print(f"🏆 Score final du Comité IA (3 Classes) : {bonnes_reponses}/{len(cube_files)} correctes ({accuracy:.1f}%)")

if __name__ == "__main__":
    run_trio_survival_prediction()