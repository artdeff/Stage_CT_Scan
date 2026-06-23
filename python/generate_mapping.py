"""
=============================================================================
MODULE : generate_mapping.py
RÔLE   : Parcourir le dossier 01_raw, lire la balise PatientID et PatientName
         dans les fichiers DICOM pour générer une table de correspondance CSV.
=============================================================================
"""

import os
from pathlib import Path
import pydicom
import pandas as pd

RAW_DIR = Path(r"..\data\01_raw")
OUTPUT_CSV = Path(r"..\data\mapping_id_nom.csv")

def build_mapping_table():
    print(f"🔍 Scan du dossier {RAW_DIR} pour créer le pont ID <-> Nom...")
    mapping_data = []

    # Parcourir les dossiers de chaque patient (qui portent l'ID numérique)
    for patient_folder in RAW_DIR.iterdir():
        if patient_folder.is_dir():
            patient_id_folder = patient_folder.name
            
            # Trouver le tout premier fichier .dcm dans ce dossier pour extraire le nom
            first_dicom = next(patient_folder.glob("**/*.dcm"), None)
            
            if first_dicom:
                try:
                    # Lire uniquement l'en-tête pour aller vite
                    dicom_data = pydicom.dcmread(first_dicom, stop_before_pixels=True)
                    
                    # Récupérer le nom inscrit dans le scanner (balise PatientName)
                    patient_name = str(getattr(dicom_data, 'PatientName', 'Inconnu'))
                    
                    # Nettoyer le format DICOM classique (ex: "DUPONT^JEAN" -> "DUPONT JEAN")
                    patient_name = patient_name.replace('^', ' ').strip()
                    
                    mapping_data.append({
                        "PatientID": str(patient_id_folder),
                        "Nom_Dicom": patient_name
                    })
                    print(f"🔗 Liaison établie : {patient_id_folder} <--> {patient_name}")
                    
                except Exception as e:
                    print(f"⚠️ Impossible de lire le fichier {first_dicom.name} : {e}")

    if mapping_data:
        df_mapping = pd.DataFrame(mapping_data)
        df_mapping.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ Table de correspondance sauvegardée avec succès dans : {OUTPUT_CSV}")
    else:
        print("❌ Aucun fichier DICOM valide trouvé pour créer la correspondance.")

if __name__ == "__main__":
    build_mapping_table()