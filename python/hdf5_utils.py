"""
=============================================================================
MODULE : hdf5_utils.py
RÔLE   : Lire les volumes CT exportés par MATLAB (.h5) et corriger les axes.
=============================================================================
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_ct_from_hdf5(filepath: str | Path, dataset_name: str = '/volume_ct'):
    """
    Charge un volume CT et ses métadonnées depuis un fichier HDF5.
    
    Args:
        filepath: Chemin vers le fichier .h5.
        dataset_name: Nom du dataset interne (défini dans MATLAB).
        
    Returns:
        volume (numpy.ndarray): Le volume 3D au format (H, W, D).
        metadata (dict): Dictionnaire contenant les métadonnées (PixelSpacing, etc.).
    """
    file_path = Path(filepath)
    
    if not file_path.exists():
        raise FileNotFoundError(f"⚠️ Fichier introuvable : {file_path}")

    with h5py.File(file_path, 'r') as f:
        # 1. Extraction du volume brut
        if dataset_name not in f:
            raise KeyError(f"⚠️ Le dataset '{dataset_name}' n'existe pas dans le fichier.")
            
        volume_raw = f[dataset_name][:]
        
        # 2. 🚨 CORRECTION DES DIMENSIONS (Le piège Fortran vs C) 🚨
        # MATLAB sauve en (H, W, D) -> h5py lit en (D, W, H)
        # On transpose pour retrouver l'ordre naturel d'une image médicale : (H, W, D)
        volume_corrected = np.transpose(volume_raw, (2, 1, 0))
        
        # 3. Extraction des métadonnées (Attributs)
        metadata = {}
        for key, value in f[dataset_name].attrs.items():
            # Conversion propre des tableaux HDF5 en listes ou scalaires Python
            if isinstance(value, np.ndarray):
                metadata[key] = value.tolist() if value.size > 1 else value.item()
            else:
                metadata[key] = value

    return volume_corrected, metadata

def plot_middle_slice(volume: np.ndarray):
    """
    Affiche la coupe axiale (Z) située au milieu du volume 3D.
    """
    # Si le volume est (H, W, D), la profondeur est le dernier axe
    depth = volume.shape[2]
    mid_z = depth // 2
    
    slice_2d = volume[:, :, mid_z]
    
    plt.figure(figsize=(6, 6))
    # cmap='gray' est crucial pour les images médicales, vmin/vmax permettent un bon contraste HU
    plt.imshow(slice_2d, cmap='gray')
    plt.title(f"Coupe axiale Z = {mid_z} / {depth-1}")
    plt.axis('off')
    plt.show()

# =========================================================================
# BLOC DE TEST (Ne s'exécute que si on lance ce script directement)
# =========================================================================
if __name__ == "__main__":
    # Chemin relatif depuis le dossier python/ vers data/02_interim/
    # (À adapter dès que tu auras ton premier vrai fichier .h5)
    test_file = Path("..") / "data" / "02_interim" / "patient_01.h5"
    
    print(f"🔍 Tentative de lecture de : {test_file.resolve()}")
    
    try:
        vol, meta = load_ct_from_hdf5(test_file)
        
        print("\n✅ Chargement réussi !")
        print(f"📏 Dimensions corrigées du volume (H, W, D) : {vol.shape}")
        print(f"📊 Type de données : {vol.dtype}")
        
        print("\n📎 Métadonnées extraites :")
        for k, v in meta.items():
            print(f"   - {k} : {v}")
            
        print("\n🖼️ Affichage de la coupe centrale...")
        plot_middle_slice(vol)
        
    except FileNotFoundError:
        print("\n⏳ Pas encore de fichier .h5 à tester. Le script est prêt !")