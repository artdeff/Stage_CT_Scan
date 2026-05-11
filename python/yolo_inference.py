"""
=============================================================================
MODULE : yolo_inference.py
RÔLE   : Préparer les coupes 3D pour YOLO et exécuter l'inférence 2D.
=============================================================================
"""

import numpy as np
import cv2
from ultralytics import YOLO

# En lui donnant ce nom, la librairie ultralytics téléchargera 
# automatiquement les poids de la dernière version (YOLO26 Nano)
model = YOLO('yolo26n.pt') 

# Plus tard, l'inférence sur ta coupe préparée ressemblera à ceci :
# results = model(slice_rgb)

def window_image_hu(image: np.ndarray, window_center: int, window_width: int) -> np.ndarray:
    """
    Applique un fenêtrage (Windowing) radiologique sur une image en Unités Hounsfield
    et la convertit en image 8-bits (0-255) compatible avec YOLO.
    
    Args:
        image: Coupe 2D en Unités Hounsfield (HU).
        window_center (WL): Centre de la fenêtre (ex: 40 pour l'abdomen, -600 pour les poumons).
        window_width (WW): Largeur de la fenêtre (ex: 400 pour l'abdomen, 1500 pour les poumons).
        
    Returns:
        Image 2D normalisée en uint8 (0 à 255).
    """
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    
    # On clip (coupe) les valeurs en dehors de la fenêtre
    windowed_img = np.clip(image, img_min, img_max)
    
    # Normalisation linéaire entre 0 et 255
    windowed_img = (windowed_img - img_min) / (img_max - img_min) * 255.0
    
    return windowed_img.astype(np.uint8)

def prepare_slice_for_yolo(slice_hu: np.ndarray, wc: int, ww: int) -> np.ndarray:
    """
    Prend une coupe brute HU, applique le fenêtrage et duplique les canaux pour YOLO.
    """
    # 1. Fenêtrage (HU -> 0-255)
    slice_8bit = window_image_hu(slice_hu, wc, ww)
    
    # 2. Duplication sur 3 canaux (H, W) -> (H, W, 3)
    slice_rgb = cv2.cvtColor(slice_8bit, cv2.COLOR_GRAY2RGB)
    
    return slice_rgb

# =========================================================================
# TEST RAPIDE
# =========================================================================
if __name__ == "__main__":
    print("✅ Fonctions de préparation YOLO chargées avec succès.")
    # On testera l'inférence complète une fois qu'on aura défini notre organe cible !