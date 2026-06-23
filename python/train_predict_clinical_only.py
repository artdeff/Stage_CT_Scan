"""
=============================================================================
MODULE : train_predict_clinical_only.py
RÔLE   : POINT 5 DU CAHIER DES CHARGES. 
         Modèle d'IA (Random Forest) utilisant UNIQUEMENT les données cliniques 
         de l'Excel pour prédire la survie (Apprentissage Supervisé).
=============================================================================
"""

import pandas as pd
import numpy as np
import unicodedata
import re
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold

# Config
TRAIN_CUBES_DIR = Path(r"..\data\03_processed\pancreas_cubes")
TEST_CUBES_DIR = Path(r"..\data\05_predicted_cubes")
EXCEL_PATH = r"..\data\PROGNOSTIC RADIOMICS DATABASE.xlsx"
MAPPING_PATH = r"..\data\mapping_id_nom.csv"

# Désactive les warnings inutiles de Pandas
pd.options.mode.chained_assignment = None

def normaliser_nom(nom):
    if pd.isna(nom): return ""
    nom_str = str(nom).upper()
    nom_sans_accents = ''.join(c for c in unicodedata.normalize('NFD', nom_str) if unicodedata.category(c) != 'Mn')
    nom_lettres = re.sub(r'[^A-Z\s]', ' ', nom_sans_accents)
    nom_lettres = nom_lettres.replace("SEGMENTARE", "").replace("SEGEMENTARE", "")
    return " ".join(sorted(nom_lettres.split()))

def preparer_donnees_cliniques():
    """Prépare le dataset 100% clinique."""
    df = pd.read_excel(EXCEL_PATH)
    df['Nom_Nettoye'] = df['Nume'].apply(normaliser_nom)
    
    colonne_temps = "PERIOADA SUPRAVIETUIRE (ZILE)"
    
    # Classes (Tertiles)
    seuil_bas = df[colonne_temps].quantile(0.33)
    seuil_haut = df[colonne_temps].quantile(0.67)
    conditions = [
        df[colonne_temps] < seuil_bas, 
        (df[colonne_temps] >= seuil_bas) & (df[colonne_temps] <= seuil_haut), 
        df[colonne_temps] > seuil_haut
    ]
    df['Label_Survie'] = np.select(conditions, [0, 1, 2])
    labels_dict = dict(zip(df['Nom_Nettoye'], df['Label_Survie']))
    
    # Sélection stricte des biomarqueurs et de la taille (AUCUNE DATE NI SURVIE)
    colonnes_a_garder = [
        'Dim 1CT', 'Dim 2CT', 'CA 19-9', 'CEA', 'Location (1-cap/', 'nvazie local'
    ]
    
    colonnes_presentes = [c for c in colonnes_a_garder if c in df.columns]
    df_filtre = df[colonnes_presentes].copy()
    df_filtre = df_filtre.select_dtypes(exclude=['datetime', 'datetime64', 'datetime64[ns]'])
    
    # Gestion des valeurs vides
    for col in df_filtre.columns:
        if df_filtre[col].dtype == 'object': 
            df_filtre[col] = df_filtre[col].fillna("Inconnu")
        else: 
            df_filtre[col] = df_filtre[col].fillna(0)
            
    df_clinique_encode = pd.get_dummies(df_filtre, drop_first=False)
    df_clinique_encode['Nom_Nettoye'] = df['Nom_Nettoye']
    
    return labels_dict, df_clinique_encode

def extraire_patients_par_dossier(dossier, df_clinique, labels_dict, is_test=False):
    """Sépare Train et Test pour avoir exactement les mêmes patients que les autres modèles."""
    X, y = [], []
    fichiers = sorted(list(dossier.glob("*.nrrd")))
    
    df_mapping = pd.read_csv(MAPPING_PATH)
    df_mapping['Nom_Nettoye'] = df_mapping['Nom_Dicom'].apply(normaliser_nom)
    df_mapping['PatientID'] = df_mapping['PatientID'].astype(str)
    dict_mapping = dict(zip(df_mapping['PatientID'], df_mapping['Nom_Nettoye']))

    for f in fichiers:
        if not is_test:
            nom_fichier = f.name.replace("cube_", "").replace("_0000.nrrd", "").replace(".nrrd", "").replace("_", " ")
            nom_nettoye = normaliser_nom(nom_fichier)
        else:
            match = re.search(r'\d+', f.name)
            patient_id = match.group() if match else ""
            nom_nettoye = dict_mapping.get(patient_id, "")
            
        if nom_nettoye in labels_dict:
            row = df_clinique[df_clinique['Nom_Nettoye'] == nom_nettoye].drop(columns=['Nom_Nettoye']).values[0]
            X.append(list(row))
            y.append(labels_dict[nom_nettoye])
            
    return np.array(X), np.array(y)

def main():
    print("🏥 Initialisation du Modèle 100% Clinique (Apprentissage Supervisé)...")
    labels_dict, df_clinique = preparer_donnees_cliniques()
    
    # Récupération de l'entraînement et du test
    X_train, y_train = extraire_patients_par_dossier(TRAIN_CUBES_DIR, df_clinique, labels_dict, is_test=False)
    X_test, y_test = extraire_patients_par_dossier(TEST_CUBES_DIR, df_clinique, labels_dict, is_test=True)
    
    print(f"📦 Entraînement sur {len(X_train)} patients.")
    print(f"📦 Test sur {len(X_test)} patients.")
    
    # Validation croisée interne
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X_train, y_train, cv=cv)
    print(f"\n📈 Précision K-Fold (Données Cliniques) : {scores.mean()*100:.1f}%")
    
    # Test final
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    bonnes_reponses = np.sum(predictions == y_test)
    
    print("="*50)
    print(f"🏆 SCORE FINAL (CLINIQUE SEULE) : {bonnes_reponses}/{len(y_test)} correctes ({(bonnes_reponses/len(y_test))*100:.1f}%)")
    print("="*50)

if __name__ == "__main__":
    main()