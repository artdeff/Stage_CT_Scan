"""
=============================================================================
MODULE : train_predict_hybrid_multimodal.py
RÔLE   : IA MULTIMODALE - Extrait 21 features des 7 modèles d'imagerie, 
         les fusionne avec les données cliniques de l'Excel (sans la survie),
         et entraîne un Random Forest pour le verdict final.
=============================================================================
"""

import os
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import unicodedata
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized, ToTensord

# Importation de nos architectures de référence
from survival_model import create_survival_classifier
from alternative_survival_model import create_alternative_survival_classifier
from third_survival_model import create_resnet_survival_classifier

pd.options.mode.chained_assignment = None  # Désactive les warnings inutiles de Pandas

# Config de base
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_DIR = Path(r"..\models\weights")
TRAIN_CUBES_DIR = Path(r"..\data\03_processed\pancreas_cubes")
TEST_CUBES_DIR = Path(r"..\data\05_predicted_cubes")
EXCEL_PATH = r"..\data\PROGNOSTIC RADIOMICS DATABASE.xlsx"
MAPPING_PATH = r"..\data\mapping_id_nom.csv"

def normaliser_nom(nom):
    if pd.isna(nom): return ""
    nom_str = str(nom).upper()
    nom_sans_accents = ''.join(c for c in unicodedata.normalize('NFD', nom_str) if unicodedata.category(c) != 'Mn')
    nom_lettres = re.sub(r'[^A-Z\s]', ' ', nom_sans_accents)
    nom_lettres = nom_lettres.replace("SEGMENTARE", "").replace("SEGEMENTARE", "")
    return " ".join(sorted(nom_lettres.split()))

"""
def preparer_donnees_cliniques():
    #Charge l'Excel, prépare les 3 classes cibles, et vectorise la clinique.
    df = pd.read_excel(EXCEL_PATH)
    df['Nom_Nettoye'] = df['Nume'].apply(normaliser_nom)
    
    colonne_temps = "PERIOADA SUPRAVIETUIRE (ZILE)"
    
    # Calcul des 3 classes (Tertiles)
    seuil_bas = df[colonne_temps].quantile(0.33)
    seuil_haut = df[colonne_temps].quantile(0.67)
    conditions = [df[colonne_temps] < seuil_bas, (df[colonne_temps] >= seuil_bas) & (df[colonne_temps] <= seuil_haut), df[colonne_temps] > seuil_haut]
    df['Label_Survie'] = np.select(conditions, [0, 1, 2])
    
    # 🌟 SÉCURITÉ CRITIQUE : On isole la cible et on supprime TOUT ce qui est lié à la survie
    labels_dict = dict(zip(df['Nom_Nettoye'], df['Label_Survie']))
    
    colonnes_a_exclure = ['Nume', 'Nom_Nettoye', colonne_temps, 'Label_Survie']
    # Si tu as d'autres colonnes comme "Statut (Vivant/Décédé)", ajoute-les ici pour ne pas que l'IA triche !
    if "STATUS" in df.columns: colonnes_a_exclure.append("STATUS") 
    
    colonnes_cliniques = [c for c in df.columns if c not in colonnes_a_exclure]
    
    # Encodage automatique (ex: transforme "Homme"/"Femme" ou les stades en 0 et 1)
    df_clinique_encode = pd.get_dummies(df[colonnes_cliniques], drop_first=True)
    df_clinique_encode['Nom_Nettoye'] = df['Nom_Nettoye']
    
    return labels_dict, df_clinique_encode
"""


def preparer_donnees_cliniques():

    #Charge l'Excel, prépare les classes, et sélectionne UNIQUEMENT les colonnes utiles."""
    df = pd.read_excel(EXCEL_PATH)
    df['Nom_Nettoye'] = df['Nume'].apply(normaliser_nom)
    
    colonne_temps = "PERIOADA SUPRAVIETUIRE (ZILE)"
    
    # Calcul des 3 classes (Tertiles)
    seuil_bas = df[colonne_temps].quantile(0.33)
    seuil_haut = df[colonne_temps].quantile(0.67)
    conditions = [
        df[colonne_temps] < seuil_bas, 
        (df[colonne_temps] >= seuil_bas) & (df[colonne_temps] <= seuil_haut), 
        df[colonne_temps] > seuil_haut
    ]
    df['Label_Survie'] = np.select(conditions, [0, 1, 2])
    labels_dict = dict(zip(df['Nom_Nettoye'], df['Label_Survie']))
    
    # 🌟 1. SÉLECTION STRICTE DES COLONNES (D'après votre image)
    # Écrivez ici le nom exact des colonnes que l'IA a le droit de lire.
    # J'ai mis les dimensions de la tumeur et les marqueurs tumoraux. 
    # Sexe, Age (Varsta), Alcool, etc., sont TOTALEMENT IGNORÉS.
    colonnes_a_garder = [
        'Dim 1CT',           # Taille 1 de la tumeur
        'Dim 2CT',           # Taille 2 de la tumeur
        'CA 19-9',           # Marqueur tumoral
        'CEA',               # Autre marqueur
        'Location (1-cap/',  # Localisation
        'nvazie local'       # Invasion locale
        # Ajoutez ou retirez les colonnes que vous voulez ici !
    ]
    
    # On s'assure que les colonnes existent bien dans le tableau
    colonnes_presentes = [c for c in colonnes_a_garder if c in df.columns]
    df_filtre = df[colonnes_presentes].copy()
    
    # 🌟 2. GESTION DES CASES VIDES (Anti-Crash)
    # Si une case est vide, on la remplit par 0 (pour les chiffres) ou "Inconnu" (pour le texte)
    for col in df_filtre.columns:
        if df_filtre[col].dtype == 'object': # Si c'est du texte
            df_filtre[col].fillna("Inconnu", inplace=True)
        else: # Si c'est un chiffre
            df_filtre[col].fillna(0, inplace=True)
            
    # 3. Encodage automatique (ex: transforme "Inconnu" ou la Localisation en 0 et 1)
    df_clinique_encode = pd.get_dummies(df_filtre, drop_first=False)
    
    # On remet le nom nettoyé pour pouvoir faire la fusion avec les images plus tard
    df_clinique_encode['Nom_Nettoye'] = df['Nom_Nettoye']
    
    return labels_dict, df_clinique_encode

def charger_les_7_modeles_d_imagerie():
    """Déploie le super-comité des 7 IA d'imagerie."""
    # 1. DenseNet
    m_dense = create_survival_classifier(spatial_dims=3, in_channels=1, num_classes=3).to(device)
    m_dense.load_state_dict(torch.load(WEIGHTS_DIR / "best_densenet_survival.pt", map_location=device, weights_only=True))
    m_dense.eval()
    
    # 2. ResNet
    m_res = create_resnet_survival_classifier(spatial_dims=3, n_input_channels=1, num_classes=3).to(device)
    m_res.load_state_dict(torch.load(WEIGHTS_DIR / "best_resnet_survival.pt", map_location=device, weights_only=True))
    m_res.eval()
    
    # 3 à 7. Les 5 Folds du Petit Poucet
    poucet_folds = []
    for fold in range(1, 6):
        m_p = create_alternative_survival_classifier(spatial_dims=3, in_channels=1, num_classes=3).to(device)
        m_p.load_state_dict(torch.load(WEIGHTS_DIR / f"best_petit_poucet_fold_{fold}.pt", map_location=device, weights_only=True))
        m_p.eval()
        poucet_folds.append(m_p)
        
    return m_dense, m_res, poucet_folds

def extraire_features_imagerie(cube_path, modeles, transforms):
    """Fait passer un cube dans les 7 IA pour obtenir 21 probabilités (features)."""
    m_dense, m_res, poucet_folds = modeles
    data = transforms({"image": str(cube_path)})
    input_tensor = data["image"].unsqueeze(0).to(device)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            # Extraction des probabilités brutes (Softmax)
            prob_dense = torch.softmax(m_dense(input_tensor), dim=1).cpu().numpy()[0]
            prob_res = torch.softmax(m_res(input_tensor), dim=1).cpu().numpy()[0]
            probs_poucet = [torch.softmax(m(input_tensor), dim=1).cpu().numpy()[0] for m in poucet_folds]
            
    # On concatène tout : 3 + 3 + (5 * 3) = 21 features
    features = list(prob_dense) + list(prob_res)
    for p in probs_poucet:
        features += list(p)
    return features

def generer_dataset_fusionne(cube_files, modeles, transforms, labels_dict, df_clinique, is_test=False):
    """Combine les 21 features d'imagerie et les variables cliniques de l'Excel."""
    features_globales = []
    y = []
    
    df_mapping = pd.read_csv(MAPPING_PATH)
    df_mapping['Nom_Nettoye'] = df_mapping['Nom_Dicom'].apply(normaliser_nom)
    df_mapping['PatientID'] = df_mapping['PatientID'].astype(str)
    dict_mapping = dict(zip(df_mapping['PatientID'], df_mapping['Nom_Nettoye']))

    for f in cube_files:
        # 1. Extraction du nom clinique du patient
        if not is_test:
            nom_fichier = f.name.replace("cube_", "").replace("_0000.nrrd", "").replace(".nrrd", "").replace("_", " ")
            nom_nettoye = normaliser_nom(nom_fichier)
        else:
            match = re.search(r'\d+', f.name)
            patient_id = match.group() if match else ""
            nom_nettoye = dict_mapping.get(patient_id, "")
            
        if nom_nettoye in labels_dict:
            # 2. Features imagerie
            feat_img = extraire_features_imagerie(f, modeles, transforms)
            # 3. Features cliniques
            row_clinique = df_clinique[df_clinique['Nom_Nettoye'] == nom_nettoye].drop(columns=['Nom_Nettoye']).values[0]
            
            # Fusion : 21 features d'image + toutes les features cliniques
            feat_totale = list(row_clinique)
            
            features_globales.append(feat_totale)
            y.append(labels_dict[nom_nettoye])
            
    return np.array(features_globales), np.array(y)

def main():
    print("⏳ Initialisation du système Hybride Multimodal...")
    labels_dict, df_clinique = preparer_donnees_cliniques()
    modeles = charger_les_7_modeles_d_imagerie()
    
    pred_transforms = Compose([
        LoadImaged(keys=["image"], reader="ITKReader"),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(64, 64, 64), mode="bilinear"),
        ToTensord(keys=["image"])
    ])
    
    # 1. CONSTRUCTION DU DATASET D'ENTRAÎNEMENT (39 Patients)
    print("📦 Extraction des caractéristiques sur la cohorte d'entraînement...")
    train_files = sorted(list(TRAIN_CUBES_DIR.glob("*.nrrd")))
    X_train, y_train = generer_dataset_fusionne(train_files, modeles, pred_transforms, labels_dict, df_clinique, is_test=False)
    
    # 2. VALIDATION CROISÉE SUR L'ENTRAÎNEMENT
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X_train, y_train, cv=cv)
    print(f"📈 Précision estimée en Cross-Validation (Clinique + Image) : {scores.mean()*100:.1f}%")
    
    # Entraînement final du Random Forest sur les 39 patients
    rf.fit(X_train, y_train)
    
    # 3. ÉVALUATION FINALE SUR LES 56 PATIENTS DE TEST
    print("\n🚀 Évaluation finale sur les 56 cubes du dossier de test...")
    test_files = sorted(list(TEST_CUBES_DIR.glob("*.nrrd")))
    X_test, y_test = generer_dataset_fusionne(test_files, modeles, pred_transforms, labels_dict, df_clinique, is_test=True)
    
    predictions_finales = rf.predict(X_test)
    
    # Calcul du score final
    bonnes_reponses = np.sum(predictions_finales == y_test)
    final_accuracy = (bonnes_reponses / len(y_test)) * 100
    
    print("="*60)
    print(f"🏆 SCORE FINAL DE L'IA HYBRIDE : {bonnes_reponses}/{len(y_test)} correctes ({final_accuracy:.1f}%)")
    print("="*60)

if __name__ == "__main__":
    main()