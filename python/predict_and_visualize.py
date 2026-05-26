"""
=============================================================================
MODULE : predict_and_visualize.py
RÔLE   : Inférence interactive. Permet à l'utilisateur de choisir un patient
         de validation et d'afficher la superposition (Vrai vs Prédit).
=============================================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from unet_segmentation import create_unet_model
from monai.inferers import sliding_window_inference

# 1. Configuration du périphérique
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chemins
MODEL_PATH = Path(r"..\models\weights\best_unet_model.pt")

def select_patient_interactive(val_dataset):
    """Affiche un menu dans le terminal pour choisir le patient."""
    print("\n" + "="*50)
    print("📂 PATIENTS DE VALIDATION DISPONIBLES :")
    print("="*50)
    
    for i, data_dict in enumerate(val_dataset):
        # On extrait juste le nom du fichier pour que ce soit lisible
        patient_name = Path(data_dict['image']).name
        print(f"  [{i}] -> {patient_name}")
        
    print("="*50)
    
    # Boucle jusqu'à ce que l'utilisateur donne une réponse valide
    while True:
        try:
            choix = input("\n👉 Tape le numéro du patient que tu souhaites analyser (ex: 0) : ")
            index = int(choix)
            if 0 <= index < len(val_dataset):
                return index
            else:
                print(f"❌ Erreur : Le numéro doit être entre 0 et {len(val_dataset)-1}.")
        except ValueError:
            print("❌ Erreur : Tu dois taper un nombre entier.")

def evaluate_and_plot():
    print(f"🔮 Initialisation de l'inférence sur : {device}")
    
    # 2. Récupération des données (Reconstruction propre)
    from monai_dataloader import IMAGES_DIR, LABELS_DIR
    
    image_files = sorted(list(IMAGES_DIR.glob("*.nrrd")))
    label_files = sorted(list(LABELS_DIR.glob("*.nrrd")))
    data_dicts = [{"image": str(img), "label": str(lbl)} for img, lbl in zip(image_files, label_files)]
    
    # On isole les patients 26 à 32 (Ceux que l'IA n'a jamais vus !)
    val_dataset = data_dicts[0:]
    
    # --- 🌟 NOUVEAU : MENU INTERACTIF ---
    selected_index = select_patient_interactive(val_dataset)
    patient_choisi = val_dataset[selected_index]
    print(f"\n⏳ Chargement du patient : {Path(patient_choisi['image']).name}...")
    
    # Transformations pour l'inférence (Aucune augmentation ici, juste de la préparation physique !)
    from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, ScaleIntensityRanged, ToTensord
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"], reader="ITKReader"),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-150, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys=["image", "label"])
    ])
    
    patient_data = val_transforms(patient_choisi)
    
    input_tensor = patient_data["image"].unsqueeze(0).to(device)
    true_label = patient_data["label"].numpy()[0]
    
    # 3. Chargement du modèle U-Net 3D
    model = create_unet_model(spatial_dims=3, in_channels=1, out_channels=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    # 4. Inférence 3D
    print("⏳ Calcul de la segmentation par l'IA (Sliding Window)...")
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            outputs = sliding_window_inference(input_tensor, (96, 96, 96), 4, model)
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()[0]

    # 5. Recherche de la coupe optimale
    pixels_par_coupe = np.sum(true_label, axis=(0, 1))
    best_z = np.argmax(pixels_par_coupe)
    print(f"🎯 Organe détecté ! Affichage de la coupe optimale : Z = {best_z}")
    
    img_slice = patient_data["image"].numpy()[0, :, :, best_z]
    true_slice = true_label[:, :, best_z]
    pred_slice = outputs[:, :, best_z]
    
    # 6. Affichage Matplotlib
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_slice.T, cmap="gray")
    plt.title("Image CT (Validation)")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(img_slice.T, cmap="gray")
    masked_true = np.ma.masked_where(true_slice.T == 0, true_slice.T)
    plt.imshow(masked_true, cmap="autumn", alpha=0.5, interpolation="none")
    plt.title("Vérité Terrain (Médecin)")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(img_slice.T, cmap="gray")
    masked_pred = np.ma.masked_where(pred_slice.T == 0, pred_slice.T)
    plt.imshow(masked_pred, cmap="winter", alpha=0.5, interpolation="none")
    plt.title("Prédiction (Notre IA)")
    plt.axis("off")
    
    plt.tight_layout()
    print("🖼️ Affichage en cours. Ferme la fenêtre pour analyser un autre patient !")
    plt.show()

if __name__ == "__main__":
    evaluate_and_plot()