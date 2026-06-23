"""
=============================================================================
MODULE : predict_survival_grand_comite.py
RÔLE   : Charger DenseNet, ResNet ET les 5 Folds du "Petit Poucet" pour un
         grand vote majoritaire à 7 voix sur les 56 cubes segmentés.
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

# Importation des 3 architectures de référence
from survival_model import create_survival_classifier  # DenseNet
from alternative_survival_model import create_alternative_survival_classifier  # Petit Poucet
from third_survival_model import create_resnet_survival_classifier  # ResNet

def get_real_clinical_labels(cube_files):
    """Traduit les IDs numériques des fichiers grâce au CSV et récupère la survie sur l'Excel."""
    excel_path = r"..\data\PROGNOSTIC RADIOMICS DATABASE.xlsx"
    mapping_path = r"..\data\mapping_id_nom.csv"
    
    df_clinique = pd.read_excel(excel_path)
    df_mapping = pd.read_csv(mapping_path)

    def normaliser_nom(nom):
        if pd.isna(nom): return ""
        nom_str = str(nom).upper()
        nom_sans_accents = ''.join(c for c in unicodedata.normalize('NFD', nom_str) if unicodedata.category(c) != 'Mn')
        nom_lettres = re.sub(r'[^A-Z\s]', ' ', nom_sans_accents)
        nom_lettres = nom_lettres.replace("SEGMENTARE", "").replace("SEGEMENTARE", "")
        return " ".join(sorted(nom_lettres.split()))

    df_clinique['Nom_Nettoye'] = df_clinique['Nume'].apply(normaliser_nom)
    df_mapping['Nom_Nettoye'] = df_mapping['Nom_Dicom'].apply(normaliser_nom)
    df_mapping['PatientID'] = df_mapping['PatientID'].astype(str)

    # Règle des 3 Classes (Tertiles)
    colonne_temps = "PERIOADA SUPRAVIETUIRE (ZILE)"
    seuil_bas = df_clinique[colonne_temps].quantile(0.33)
    seuil_haut = df_clinique[colonne_temps].quantile(0.67)
    
    conditions = [
        df_clinique[colonne_temps] < seuil_bas,
        (df_clinique[colonne_temps] >= seuil_bas) & (df_clinique[colonne_temps] <= seuil_haut),
        df_clinique[colonne_temps] > seuil_haut
    ]
    df_clinique['Label_Survie'] = np.select(conditions, [0, 1, 2])
    
    df_fusion = pd.merge(df_mapping, df_clinique, on='Nom_Nettoye', how='inner')
    dict_id_vers_survie = dict(zip(df_fusion['PatientID'], df_fusion['Label_Survie']))
    
    labels_finaux = {}
    for f in cube_files:
        match = re.search(r'\d+', f.name)
        if match:
            patient_id = match.group()
            if patient_id in dict_id_vers_survie:
                labels_finaux[f.name] = int(dict_id_vers_survie[patient_id])
            else:
                labels_finaux[f.name] = 0
        else:
            labels_finaux[f.name] = 0
            
    return labels_finaux

def run_grand_comite_prediction():
    print("⏳ Initialisation du Grand Comité des 7 modèles IA...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    WEIGHTS_DIR = Path(r"..\models\weights")
    #CUBES_DIR = Path(r"..\data\05_predicted_cubes") # Vos 56 cubes rognés !
    CUBES_DIR = Path(r"..\data\03_processed\pancreas_cubes")

    cube_files = sorted(list(CUBES_DIR.glob("*.nrrd")))
    print(f"📦 Nombre de cubes détectés pour le test : {len(cube_files)}\n")
    
    if len(cube_files) == 0:
        print(f"⚠️ Aucun fichier trouvé dans {CUBES_DIR}")
        return

    vrais_labels_dict = get_real_clinical_labels(cube_files)

    # 1. Chargement de DenseNet (3 classes)
    model_dense = create_survival_classifier(spatial_dims=3, in_channels=1, num_classes=3).to(device)
    model_dense.load_state_dict(torch.load(WEIGHTS_DIR / "best_densenet_survival.pt", map_location=device, weights_only=True))
    model_dense.eval()
    
    # 2. Chargement de ResNet (3 classes)
    model_res = create_resnet_survival_classifier(spatial_dims=3, n_input_channels=1, num_classes=3).to(device)
    model_res.load_state_dict(torch.load(WEIGHTS_DIR / "best_resnet_survival.pt", map_location=device, weights_only=True))
    model_res.eval()
    
    # 3. Chargement dynamique des 5 Folds du Petit Poucet
    poucet_folds = []
    for fold in range(1, 6):
        model_p = create_alternative_survival_classifier(spatial_dims=3, in_channels=1, num_classes=3).to(device)
        model_p.load_state_dict(torch.load(WEIGHTS_DIR / f"best_petit_poucet_fold_{fold}.pt", map_location=device, weights_only=True))
        model_p.eval()
        poucet_folds.append(model_p)
    
    print("✅ Le Grand Comité (7 IA) est déployé et prêt !\n")
    
    pred_transforms = Compose([
        LoadImaged(keys=["image"], reader="ITKReader"),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(64, 64, 64), mode="bilinear"),
        ToTensord(keys=["image"])
    ])
    
    # En-tête de tableau adapté pour 7 modèles
    print(f"{'Patient':<25}| {'Dense':<5} | {'Res':<3} | {'F1':<2} {'F2':<2} {'F3':<2} {'F4':<2} {'F5':<2} || {'Vote IA':<8} | {'Vrai':<8} | {'Résultat':<10}")
    print("-" * 110)
    
    mapping = {0: "Courte", 1: "Moyenne", 2: "Longue"}
    bonnes_reponses = 0
    
    with torch.no_grad():
        for cube_path in cube_files:
            
            patient_name = cube_path.name.replace("cube_raw_", "").replace(".nrrd", "")
            vraie_survie_num = vrais_labels_dict[cube_path.name]
            txt_vrai = mapping[vraie_survie_num]
            
            try:
                data = pred_transforms({"image": str(cube_path)})
                input_tensor = data["image"].unsqueeze(0).to(device)
                
                with torch.amp.autocast('cuda'):
                    # Prédictions des deux gros modèles globals
                    pred_dense = torch.argmax(model_dense(input_tensor), dim=1).item()
                    pred_res = torch.argmax(model_res(input_tensor), dim=1).item()
                    
                    # Prédictions des 5 folds experts
                    preds_poucet = [torch.argmax(m(input_tensor), dim=1).item() for m in poucet_folds]
                
                # Fusion de toutes les voix (2 + 5 = 7 votes au total)
                tous_les_votes = [pred_dense, pred_res] + preds_poucet
                vote_majoritaire = Counter(tous_les_votes).most_common(1)[0][0]
                
                # Traduction en première lettre pour l'affichage compact
                c_dense = mapping[pred_dense][0]
                c_res = mapping[pred_res][0]
                c_folds = [mapping[v][0] for v in preds_poucet]
                
                txt_final = mapping[vote_majoritaire]
                
                if vote_majoritaire == vraie_survie_num:
                    statut = "✅ SUCCÈS"
                    bonnes_reponses += 1
                else:
                    statut = "❌ ÉCHEC"
                    
                # Affichage de la ligne du patient
                print(f"ID {patient_name[:25]:<25} | {c_dense:<5} | {c_res:<3} | {c_folds[0]:<2} {c_folds[1]:<2} {c_folds[2]:<2} {c_folds[3]:<2} {c_folds[4]:<2} || {txt_final:<8} | {txt_vrai:<8} | {statut}")
                
            except Exception as e:
                print(f"ID {patient_name[:25]:<25} | ❌ Erreur : {e}")  

    print("-" * 110)
    accuracy = (bonnes_reponses / len(cube_files)) * 100
    print(f"🏆 Score final du Grand Comité Élégant : {bonnes_reponses}/{len(cube_files)} correctes ({accuracy:.1f}%)")

if __name__ == "__main__":
    run_grand_comite_prediction()