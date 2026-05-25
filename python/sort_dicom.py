"""
=============================================================================
MODULE : sort_dicom.py
RÔLE   : Trier un export CD hospitalier vers une arborescence propre pour la R&D
=============================================================================
"""

import os
import shutil
import pydicom
from pathlib import Path

# 1. Définition des chemins (À ADAPTER SELON TON PC)
# Chemin vers le dossier "Raw Database" que tu viens de recevoir
SOURCE_DIR = Path(r"C:\Users\Arthur\Documents\Travail\Polytech\4A\Stage\CT Database\Raw Database")
# Chemin vers notre dossier de projet propre
DEST_DIR = Path(r"..\data\01_raw")

def sort_hospital_dicoms(source_dir: Path, dest_dir: Path):
    print(f"🔍 Scan du répertoire source : {source_dir}")
    
    # Parcourt tous les fichiers, y compris dans les sous-dossiers
    for root, _, files in os.walk(source_dir):
        for file in files:
            # On ignore les fichiers non-images classiques
            if file in ['DICOMDIR', 'Autorun.inf', 'AUTORUN.INF'] or file.endswith(('.txt', '.html', '.css')):
                continue
                
            file_path = Path(root) / file
            
            try:
                # Lecture de l'en-tête DICOM (stop_before_pixels=True pour aller très vite)
                dicom_data = pydicom.dcmread(file_path, stop_before_pixels=True)
                
                # Extraction des métadonnées (avec des valeurs par défaut si absentes)
                patient_id = str(getattr(dicom_data, 'PatientID', 'Unknown_Patient'))
                # On nettoie le nom du patient des caractères bizarres
                patient_id = "".join([c for c in patient_id if c.isalnum() or c in ('_', '-')])
                
                series_desc = str(getattr(dicom_data, 'SeriesDescription', 'Unknown_Series'))
                series_desc = "".join([c for c in series_desc if c.isalnum() or c in ('_', '-')])
                
                # 2. Création du nouveau chemin
                new_folder = dest_dir / patient_id / series_desc
                new_folder.mkdir(parents=True, exist_ok=True)
                
                # Ajout de l'extension .dcm et formatage du nom (ex: IM000031.dcm)
                new_file_path = new_folder / f"{file}.dcm"
                
                # 3. Copie du fichier
                if not new_file_path.exists():
                    shutil.copy2(file_path, new_file_path)
                    print(f"✅ Copié : {patient_id} / {series_desc} / {file}.dcm")
                    
            except pydicom.errors.InvalidDicomError:
                # Ce n'est pas un fichier DICOM valide, on l'ignore
                pass
            except Exception as e:
                print(f"⚠️ Erreur sur {file_path.name} : {e}")

if __name__ == "__main__":
    # S'assure que le dossier de destination existe
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    sort_hospital_dicoms(SOURCE_DIR, DEST_DIR)
    print("\n🎉 Tri terminé ! Tes données sont prêtes pour MATLAB.")