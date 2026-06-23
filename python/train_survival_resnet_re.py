"""
=============================================================================
MODULE : train_survival_resnet.py
RÔLE   : Charger les VRAIS volumes NRRD bruts de l'équipe médicale,
         leur associer les étiquettes, et entraîner le classifieur ResNet3D.
=============================================================================
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd


from monai.data import Dataset, DataLoader
from monai.utils import set_determinism
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized, RandRotated, RandZoomd, ToTensord

# Importation du troisième modèle de décision
from third_survival_model import create_resnet_survival_classifier

# 1. Configuration de base
set_determinism(seed=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CUBES_DIR = Path(r"..\data\03_processed\pancreas_cubes")
WEIGHTS_DIR = Path(r"..\models\weights")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
best_model_path = WEIGHTS_DIR / "best_resnet_survival.pt"


def get_real_clinical_labels(cube_files):
    excel_path = r"..\data\PROGNOSTIC RADIOMICS DATABASE.xlsx"
    df_clinique = pd.read_excel(excel_path)
    
    import unicodedata
    import re

    def normaliser_nom(nom):
        if pd.isna(nom): return ""
        nom_str = str(nom).upper()
        # Enlever les accents
        nom_sans_accents = ''.join(c for c in unicodedata.normalize('NFD', nom_str) if unicodedata.category(c) != 'Mn')
        # Garder uniquement les lettres
        nom_lettres = re.sub(r'[^A-Z\s]', ' ', nom_sans_accents)
        # Supprimer le mot "SEGMENTARE" (et ses fautes de frappe)
        nom_lettres = nom_lettres.replace("SEGMENTARE", "").replace("SEGEMENTARE", "")
        # Trier par ordre alphabétique
        return " ".join(sorted(nom_lettres.split()))

    df_clinique['Nom_Nettoye'] = df_clinique['Nume'].apply(normaliser_nom)
    
    colonne_temps = "PERIOADA SUPRAVIETUIRE (ZILE)"
    
    # Calcul des deux seuils (33% et 67%) pour diviser en 3 groupes égaux
    seuil_bas = df_clinique[colonne_temps].quantile(0.33)
    seuil_haut = df_clinique[colonne_temps].quantile(0.67)
    
    print(f"📊 Seuils de survie - Bas (<{seuil_bas:.1f} j), Moyen ({seuil_bas:.1f}-{seuil_haut:.1f} j), Haut (>{seuil_haut:.1f} j)")
    
    # Création des 3 classes : 0 (Courte), 1 (Moyenne), 2 (Longue)
    conditions = [
        df_clinique[colonne_temps] < seuil_bas,
        (df_clinique[colonne_temps] >= seuil_bas) & (df_clinique[colonne_temps] <= seuil_haut),
        df_clinique[colonne_temps] > seuil_haut
    ]
    valeurs = [0, 1, 2]
    df_clinique['Label_Survie'] = np.select(conditions, valeurs)
    
    dict_clinique = dict(zip(df_clinique['Nom_Nettoye'], df_clinique['Label_Survie']))
    
    labels_finaux = {}
    succes_fusion = 0
    
    for f in cube_files:
        # On extrait le texte du nom de fichier
        nom_fichier = f.name.replace("cube_", "").replace("_0000.nrrd", "").replace(".nrrd", "").replace("_", " ")
        nom_nettoye = normaliser_nom(nom_fichier)
        
        if nom_nettoye in dict_clinique:
            labels_finaux[f.name] = int(dict_clinique[nom_nettoye])
            succes_fusion += 1
        else:
            print(f"⚠️ Patient introuvable : '{nom_fichier}' -> (Lu par l'IA comme: '{nom_nettoye}')")
            labels_finaux[f.name] = 0
            
    print(f"🔗 Fusion réussie pour {succes_fusion} patients sur {len(cube_files)} présents.")
    return labels_finaux

def train_resnet_pipeline():
    print(f"🖥️ Entraînement ResNet configuré sur : {device}")
    
    # 🌟 RECHERCHE DES VRAIS FICHIERS .nrrd BRUTS
    cube_files = sorted(list(CUBES_DIR.glob("*.nrrd")))
    if not cube_files:
        print(f"❌ Aucun fichier .nrrd trouvé dans {CUBES_DIR}. Vérifiez l'emplacement de vos 56 cubes réels !")
        return
        
    labels_dict = get_real_clinical_labels(cube_files)
    data_dicts = [{"image": str(f), "label": labels_dict[f.name]} for f in cube_files]
    print(f"📦 Nombre de cubes réels détectés : {len(cube_files)}")
    
    # 2. Pipeline de transformations pour fichiers NRRD
    train_transforms = Compose([
        LoadImaged(keys=["image"], reader="ITKReader"), # 🌟 Alignement sur l'ITKReader de référence
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(64, 64, 64), mode="bilinear"),
        RandRotated(keys=["image"], range_x=0.2, range_y=0.2, range_z=0.2, prob=0.5, mode="bilinear"),
        RandZoomd(keys=["image"], min_zoom=0.9, max_zoom=1.1, prob=0.5, mode="bilinear"),
        ToTensord(keys=["image", "label"])
    ])

    dataset = Dataset(data=data_dicts, transform=train_transforms)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    # 3. Initialisation du modèle ResNet-10 3D
    model = create_resnet_survival_classifier(spatial_dims=3, n_input_channels=1, num_classes=3).to(device)
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    EPOCHS = 40
    best_loss = float("inf")
    
    print(f"\n🚀 Lancement de l'entraînement réel (ResNet3D)...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        step = 0
        
        for batch_data in loader:
            step += 1
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            
            optimizer.zero_grad()
            
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            
        epoch_loss /= step
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss Classification (ResNet) : {epoch_loss:.4f}")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 Poids sauvegardés sous : {best_model_path.name} (Loss: {best_loss:.4f})")
            
    print(f"\n🎉 Entraînement fini ! Fichier disponible : {best_model_path}")

if __name__ == "__main__":
    train_resnet_pipeline()