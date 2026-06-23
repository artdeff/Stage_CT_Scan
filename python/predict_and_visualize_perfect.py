"""
=============================================================================
MODULE : predict_and_visualize_perfect.py
RÔLE   : Inférence et visualisation avancée avec pipeline MONAI.
         Superpose les masques multi-classes (Médecin vs IA) avec des couleurs
         distinctes et professionnelles directement sur le scanner.
=============================================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path

from unet_segmentation import create_unet_model
from monai.inferers import sliding_window_inference

# Configuration du périphérique
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path(r"..\models\weights\best_unet_model.pt")

def select_patient_interactive(val_dataset):
    """Menu interactif pour choisir proprement le patient de validation."""
    print("\n" + "="*50)
    print("📂 PATIENTS DE VALIDATION DISPONIBLES :")
    print("="*50)
    for i, data_dict in enumerate(val_dataset):
        print(f"  [{i}] -> {Path(data_dict['image']).name}")
    print("="*50)
    
    while True:
        try:
            choix = input("\n👉 Tape le numéro du patient à analyser (ex: 0) : ")
            index = int(choix)
            if 0 <= index < len(val_dataset):
                return index
            print(f"❌ Index invalide. Choisir entre 0 et {len(val_dataset)-1}.")
        except ValueError:
            print("❌ Erreur : Entre un nombre entier.")

def colorier_masque_multiclass(slice_masque):
    """
    Transforme un masque 2D contenant des classes (0, 1, 2, 3) en une image RGBA 
    avec des couleurs distinctes et une transparence (alpha) pour la superposition.
    """
    # Définition des couleurs professionnelles pour chaque classe
    # Classe 0 = Fond (Transparent)
    # Classe 1 = Vert (ex: Pancréas sain)
    # Classe 2 = Rouge (ex: Tumeur)
    # Classe 3 = Bleu ou Jaune (ex: Invasion / Vaisseaux)
    couleurs = {
        1: [0.0, 0.8, 0.0, 0.4],  # Vert, Opacité 40%
        2: [0.9, 0.0, 0.0, 0.5],  # Rouge, Opacité 50%
        3: [0.0, 0.4, 0.9, 0.4]   # Bleu, Opacité 40%
    }
    
    # Création d'une image vide RGBA (Hauteur, Largeur, 4 canaux)
    rgba_out = np.zeros((*slice_masque.shape, 4))
    
    # On applique la couleur correspondante pour chaque classe présente
    for classe_id, rgba in couleurs.items():
        rgba_out[slice_masque == classe_id] = rgba
        
    return rgba_out

def evaluate_and_plot():
    print(f"🔮 Initialisation de l'affichage expert sur : {device}")
    
    # 1. Récupération des données via votre dataloader MONAI
    from monai_dataloader import IMAGES_DIR, LABELS_DIR
    
    image_files = sorted(list(IMAGES_DIR.glob("*.nrrd")))
    label_files = sorted(list(LABELS_DIR.glob("*.nrrd")))
    data_dicts = [{"image": str(img), "label": str(lbl)} for img, lbl in zip(image_files, label_files)]
    
    val_dataset = data_dicts[0:]
    selected_index = select_patient_interactive(val_dataset)
    patient_choisi = val_dataset[selected_index]
    
    print(f"\n⏳ Pipeline MONAI : Chargement et normalisation de {Path(patient_choisi['image']).name}...")
    
    # Vos transformations MONAI parfaites (Origine, Spacing, Échelle d'intensité)
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
    
    # 2. Chargement du modèle U-Net 3D
    model = create_unet_model(spatial_dims=3, in_channels=1, out_channels=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    # 3. Inférence par fenêtre glissante (Sliding Window)
    print("🧠 Calcul de la segmentation par l'IA...")
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            outputs = sliding_window_inference(input_tensor, (96, 96, 96), 4, model)
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()[0]

    # 4. Recherche de la coupe Z optimale (Celle contenant le plus de pixels segmentés par le médecin)
    pixels_par_coupe = np.sum(true_label, axis=(0, 1))
    best_z = np.argmax(pixels_par_coupe)
    print(f"🎯 Coupe optimale identifiée : Z = {best_z}")
    
    # Extraction des coupes 2D correspondantes
    img_slice = patient_data["image"].numpy()[0, :, :, best_z]
    true_slice = true_label[:, :, best_z]
    pred_slice = outputs[:, :, best_z]
    
    # Transformation des masques d'index (0,1,2,3) en images colorées RGBA translucides
    overlay_vrai = colorier_masque_multiclass(true_slice)
    overlay_pred = colorier_masque_multiclass(pred_slice)
    
    # 5. Affichage Matplotlib Haute Qualité
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Analyse Expert U-Net - {Path(patient_choisi['image']).name} (Coupe Z={best_z})", fontsize=14, fontweight='bold')
    
    # Sous-intrigue 1 : Le Scanner Brut et Propre
    axes[0].imshow(img_slice.T, cmap="gray", origin="lower")
    axes[0].set_title("1. Scanner CT Normalisé", fontweight='semibold')
    axes[0].axis("off")
    
    # Sous-intrigue 2 : Vérité Terrain Médecin (Superposée)
    axes[1].imshow(img_slice.T, cmap="gray", origin="lower")
    axes[1].imshow(overlay_vrai.transpose(1, 0, 2), origin="lower", interpolation="none")
    axes[1].set_title("2. Vérité Terrain (Médecin)", fontweight='semibold')
    axes[1].axis("off")
    
    # Sous-intrigue 3 : Prédiction du Modèle U-Net (Superposée)
    axes[2].imshow(img_slice.T, cmap="gray", origin="lower")
    axes[2].imshow(overlay_pred.transpose(1, 0, 2), origin="lower", interpolation="none")
    axes[2].set_title("3. Prédiction (Notre U-Net 3D)", fontweight='semibold')
    axes[2].axis("off")
    
    # --- NOUVELLE LÉGENDE COLORÉE ---
    # On crée de vraies "pastilles" (patches) graphiques avec les mêmes couleurs que les masques
    legend_elements = [
        mpatches.Patch(facecolor=[0.0, 0.8, 0.0, 0.5], edgecolor='black', label='Classe 1 (Pancréas)'),
        mpatches.Patch(facecolor=[0.9, 0.0, 0.0, 0.5], edgecolor='black', label='Classe 2 (Tumeur)'),
        mpatches.Patch(facecolor=[0.0, 0.4, 0.9, 0.5], edgecolor='black', label='Classe 3 (Invasion / Autre)')
    ]
    
    # Ajout de la légende à la figure (centrée en bas)
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01), fontsize=12, framealpha=0.9)

    # L'option rect=[0, 0.08, 1, 1] permet de remonter légèrement les images 
    # pour laisser de la place à la légende en bas sans qu'elle n'écrase le scanner.
    plt.tight_layout(rect=[0, 0.08, 1, 1]) 
    
    print("🖼️ Fenêtre graphique affichée. Analyse les alignements !")
    plt.show()

if __name__ == "__main__":
    evaluate_and_plot()