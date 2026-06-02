"""
=============================================================================
MODULE : load_clinical_data.py
RÔLE   : Charger le fichier Excel, binariser la survie (Target pour l'IA)
         et nettoyer les variables cliniques pour les points 5 et 6.
=============================================================================
"""

import pandas as pd
from pathlib import Path

def parse_clinical_excel(excel_path, survival_threshold_months=12):
    print(f"📊 Chargement du fichier clinique : {excel_path}")
    
    # 1. Lecture de l'Excel
    df = pd.read_excel(excel_path)
    
    # 2. Création du dictionnaire pour l'IA Imagerie (Point 4)
    # On suppose que l'Excel a une colonne 'Patient_ID' et une colonne 'Survival_Months'
    # À AJUSTER dès que tu as le fichier visuel sous les yeux !
    clinical_dict = {}
    
    for _, row in df.iterrows():
        patient_id = str(row['Patient_ID']) # ex: "raw_14402125"
        survival_time = row['Survival_Months']
        
        # Binarisation de la survie (Seuil court terme vs long terme)
        label = 1 if survival_time >= survival_threshold_months else 0
        
        # On peut aussi stocker les variables cliniques (Point 5 & 6)
        # Exemple: Âge, Genre (qu'on transforme en chiffres/catégories)
        clinical_features = [
            float(row['Age']),
            1 if row['Gender'].lower() == 'm' else 0,
            float(row['Tumor_Size_mm'])
        ]
        
        clinical_dict[patient_id] = {
            "label": label,
            "features": clinical_features
        }
    
    print(f"✅ Analyse terminée. {len(clinical_dict)} patients chargés.")
    return clinical_dict

if __name__ == "__main__":
    # Ce script sera testé dès que ton fichier Excel sera dans ton dossier data/
    EXCEL_PATH = Path(r"..\data\01_raw\clinical_data.xlsx")
    # clinical_data = parse_clinical_excel(EXCEL_PATH)