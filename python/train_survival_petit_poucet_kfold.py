"""
=============================================================================
MODULE : train_survival_petit_poucet_kfold.py
RÔLE   : Entraîner l'IA légère "Petit Poucet" avec Data Augmentation 
         et validation croisée (K-Fold à 5 manches) sur 3 classes.
=============================================================================
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import unicodedata
import re
from sklearn.model_selection import KFold

from monai.data import Dataset, DataLoader
from monai.utils import set_determinism
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Resized, 
    RandRotated, RandZoomd, ToTensord,
    RandFlipd, RandGaussianNoised, RandAdjustContrastd
)

# Importation du modèle Petit Poucet
from alternative_survival_model import create_alternative_survival_classifier

# 1. Configuration de base
set_determinism(seed=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CUBES_DIR = Path(r"..\data\03_processed\pancreas_cubes")
WEIGHTS_DIR = Path(r"..\models\weights")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

def get_real_clinical_labels(cube_files):
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
    seuil_bas = df_clinique[colonne_temps].quantile(0.33)
    seuil_haut = df_clinique[colonne_temps].quantile(0.67)
    
    print(f"📊 Seuils : Courte (<{seuil_bas:.1f}j), Moyenne, Longue (>{seuil_haut:.1f}j)")
    
    conditions = [
        df_clinique[colonne_temps] < seuil_bas,
        (df_clinique[colonne_temps] >= seuil_bas) & (df_clinique[colonne_temps] <= seuil_haut),
        df_clinique[colonne_temps] > seuil_haut
    ]
    df_clinique['Label_Survie'] = np.select(conditions, [0, 1, 2])
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

def train_kfold_pipeline():
    print(f"🖥️ Entraînement K-Fold configuré sur : {device}")
    
    cube_files = sorted(list(CUBES_DIR.glob("*.nrrd")))
    if not cube_files:
        print("❌ Aucun cube trouvé !")
        return
        
    labels_dict = get_real_clinical_labels(cube_files)
    data_dicts = [{"image": str(f), "label": labels_dict[f.name]} for f in cube_files]
    print(f"📦 Nombre de cubes pour le K-Fold : {len(cube_files)}")

    # Data Augmentation Extrême
    train_transforms = Compose([
        LoadImaged(keys=["image"], reader="ITKReader"),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(64, 64, 64), mode="bilinear"),
        RandRotated(keys=["image"], range_x=0.4, range_y=0.4, range_z=0.4, prob=0.5, mode="bilinear"),
        RandZoomd(keys=["image"], min_zoom=0.8, max_zoom=1.2, prob=0.5, mode="bilinear"),
        RandFlipd(keys=["image"], spatial_axis=[0, 1, 2], prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.1),
        RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.5, 2.0)),
        ToTensord(keys=["image", "label"])
    ])

    data_dicts_np = np.array(data_dicts)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    EPOCHS = 100
    loss_function = nn.CrossEntropyLoss()
    
    print("\n🚀 Lancement du K-Fold Cross Validation (5 Modèles)...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(data_dicts_np)):
        print(f"\n" + "="*40)
        print(f"🏆 ENTRAÎNEMENT DU FOLD {fold + 1} / 5")
        print(f"Patients entraînement: {len(train_idx)} | Patients validation cachés: {len(val_idx)}")
        print("="*40)
        
        train_data = data_dicts_np[train_idx].tolist()
        train_dataset = Dataset(data=train_data, transform=train_transforms)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
        
        # Instanciation d'un modèle NEUF pour chaque Fold avec num_classes=3
        model = create_alternative_survival_classifier(spatial_dims=3, in_channels=1, num_classes=3).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        
        best_loss = float("inf")
        nom_fichier_sauvegarde = WEIGHTS_DIR / f"best_petit_poucet_fold_{fold+1}.pt"
        
        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0
            step = 0
            
            for batch_data in train_loader:
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
            
            # Affichage allégé pour ne pas spammer la console (toutes les 10 epochs)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Fold {fold+1} | Epoch [{epoch+1}/{EPOCHS}] - Loss : {epoch_loss:.4f}")
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), nom_fichier_sauvegarde)
                
        print(f"💾 Fin du Fold {fold+1}. Meilleur modèle sauvegardé: {nom_fichier_sauvegarde.name}")

    print("\n🎉 K-Fold terminé ! 5 cerveaux experts sont sauvegardés.")

if __name__ == "__main__":
    train_kfold_pipeline()