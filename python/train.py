"""
=============================================================================
MODULE : train.py
RÔLE   : Boucle principale d'entraînement du U-Net 3D (MONAI) avec CUDA.
=============================================================================
"""

import os
from pathlib import Path
from xml.parsers.expat import model
import torch
from torch.cuda.amp import autocast, GradScaler

# Importation de nos propres modules
from unet_segmentation import create_unet_model
from monai_dataloader import prepare_dataloaders

# Importations MONAI pour les fonctions de perte et métriques médicales
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism

# 1. Configuration et reproductibilité
set_determinism(seed=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chemins de sauvegarde
WEIGHTS_DIR = Path(r"..\models\weights")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
best_model_path = WEIGHTS_DIR / "best_unet_model.pt"

# 2. Paramètres d'entraînement
EPOCHS = 50
LEARNING_RATE = 2e-4
OUT_CHANNELS = 4  # 🌟 Nos 4 classes trouvées lors du contrôle qualité !

def train_pipeline():
    print(f"🖥️ Entraînement configuré sur le périphérique : {device}")
    
    # --- Chargement des données ---
    train_loader = prepare_dataloaders()
    
    # --- Initialisation du modèle U-Net (MONAI) ---
    model = create_unet_model(spatial_dims=3, in_channels=1, out_channels=OUT_CHANNELS)
    model = model.to(device)
    
    # --- Fonction de perte et Optimiseur ---
    # DiceCELoss combine la Dice Loss (parfaite pour les classes déséquilibrées comme la tumeur)
    # et la CrossEntropy (parfaite pour stabiliser les pixels du fond)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # --- Outils pour la Précision Mixte (Économie VRAM pour ta 5070) ---
    #scaler = GradScaler()
    scaler = torch.amp.GradScaler('cuda')
    
    # --- Métrique d'évaluation ---
    # Le score de Dice va de 0 (gros échec) à 1 (segmentation parfaite du radiologue)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    best_loss = float("inf")
    
    print("\n🚀 Lancement de l'entraînement...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        step = 0
        
        for batch_data in train_loader:
            step += 1
            # Envoi des données sur la carte graphique
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device).long()

            # --- 🕵️‍♂️ BLOC DE DÉBOGAGE (À AJOUTER) ---
            if step == 1 and epoch == 0:
                print(f"\n🔍 DÉBOGAGE BATCH 1 :")
                print(f"Forme Inputs : {inputs.shape}")
                print(f"Forme Labels : {labels.shape}")
                print(f"Type Labels : {labels.dtype}")
                print(f"Valeurs uniques dans Labels : {torch.unique(labels)}")
                print(f"Min Label : {labels.min().item()} | Max Label : {labels.max().item()}\n")
            # ----------------------------------------
            
            optimizer.zero_grad()
            
            # 🌟 Utilisation de l'autocast pour activer le calcul en Float16 (Moins de VRAM, plus rapide)
            # Avant : with autocast():
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            
            # Rétropropagation du gradient sécurisée par le scaler AMP
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
        epoch_loss /= step
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss moyenne : {epoch_loss:.4f}")
        
        # --- Sauvegarde du meilleur modèle ---
        # Ici on sauvegarde dès que la Loss diminue, on affinera avec la validation plus tard
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 Nouveau meilleur modèle sauvegardé ! (Loss: {best_loss:.4f})")
            
    print(f"\n🎉 Entraînement terminé ! Le meilleur modèle est disponible ici : {best_model_path.resolve()}")

if __name__ == "__main__":
    train_pipeline()