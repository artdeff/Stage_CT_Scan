"""
=============================================================================
MODULE : convert_raw_to_nrrd.py
RÔLE   : Parcourir automatiquement les dossiers DICOM bruts, sélectionner 
         la meilleure série (veineuse/axiale) via un système de mots-clés 
         robuste, et l'exporter proprement en volume .nrrd.
=============================================================================
"""

import os
from pathlib import Path
from monai.transforms import LoadImage
import SimpleITK as sitk  #  Avec un S et un ITK majuscules !

# Configuration des chemins
RAW_DIR = Path(r"..\data\01_raw")
INTERIM_DIR = Path(r"..\data\02_interim")
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

def select_best_series(series_paths):
    """
    Algorithme de sélection intelligent basé sur le nom des dossiers 
    et le volume de fichiers.
    """
    # 🌟 Étape 1 : Mots-clés prioritaires pour la phase VEINEUSE (notre cible IA)
    VEINOUS_KEYWORDS = ["VENOS", "VENOUS", "AXAPART", "AXC2", "PORTAL"]
    for s_path in series_paths:
        name_upper = s_path.name.upper()
        if any(kw in name_upper for kw in VEINOUS_KEYWORDS):
            return s_path
            
    # 🌟 Étape 2 : Mots-clés secondaires (si pas de phase veineuse explicite trouvée)
    SECONDARY_KEYWORDS = ["AX", "ARTERIAL", "ART", "125MM"]
    for s_path in series_paths:
        name_upper = s_path.name.upper()
        # On s'assure d'éviter les reconstructions Coronales ou Sagittales qui tromperaient l'IA
        if any(kw in name_upper for kw in SECONDARY_KEYWORDS) and "COR" not in name_upper and "SAG" not in name_upper:
            return s_path
            
    # 🌟 Étape 3 : Système de secours ultime (Si aucun mot-clé ne match)
    # On compte le nombre de fichiers .dcm dans chaque sous-dossier et on prend le plus lourd
    print("    🔍 Aucun mot-clé standard trouvé. Sélection par volume de fichiers...")
    best_path = None
    max_files = 0
    
    for s_path in series_paths:
        # Compte le nombre de fichiers (fichiers dcm ou sans extension)
        file_count = len([f for f in s_path.iterdir() if f.is_file()])
        if file_count > max_files:
            max_files = file_count
            best_path = s_path
            
    # On ne valide le secours que s'il y a un vrai volume (au moins 30 coupes)
    if max_files >= 30:
        return best_path
        
    return None

def auto_convert_dataset():
    print(f"🚀 Début de la conversion de la base brute (Version Robuste)...")
    print(f"📂 Source : {RAW_DIR.resolve()}")
    print(f"📂 Destination : {INTERIM_DIR.resolve()}\n")
    
    loader = LoadImage(reader="ITKReader", image_only=False)
    success_count = 0
    
    patients = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    
    for p_dir in patients:
        print(f"👤 Analyse du Patient {p_dir.name}...")
        
        # Récupérer tous les sous-dossiers du patient
        series = [d for d in p_dir.iterdir() if d.is_dir()]
        
        # Appel de notre sélectionneur intelligent
        best_series = select_best_series(series)
                    
        if best_series:
            print(f"  🎯 Série sélectionnée : {best_series.name}")
            output_filename = INTERIM_DIR / f"raw_{p_dir.name}.nrrd"
            
            try:
                # Lecture des fichiers DICOM et écriture au format NRRD
                sitk_image = sitk.ReadImage(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(best_series)))
                sitk.WriteImage(sitk_image, str(output_filename))
                print(f"  ✅ Export réussi -> {output_filename.name}")
                success_count += 1
                
            except Exception as e:
                print(f"  ❌ Échec de la conversion pour cette série. Erreur : {e}")
        else:
            print(f"  ⚠️ Ignoré : Aucune série 3D exploitable trouvée.")
            
        print("-" * 50)

    print(f"\n🎉 Conversion terminée ! {success_count}/{len(patients)} patients exportés dans 02_interim.")

if __name__ == "__main__":
    auto_convert_dataset()