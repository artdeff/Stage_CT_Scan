import pydicom
import matplotlib.pyplot as plt
import numpy as np

def load_and_show_dicom(filepath):
    """
    Charge un fichier DICOM et l'affiche à l'écran.
    """
    print(f"Chargement du fichier : {filepath}...")
    
    try:
        # Lecture du fichier
        dicom_data = pydicom.dcmread(filepath)
        
        # Extraction de la matrice de pixels
        image_array = dicom_data.pixel_array
        
        print(f"Dimensions de l'image : {image_array.shape}")
        
        # Affichage avec matplotlib
        plt.imshow(image_array, cmap='gray')
        plt.title(f"Patient: {dicom_data.get('PatientID', 'Inconnu')}")
        plt.axis('off')
        plt.show()
        
        return image_array
        
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")
        return None

# Test rapide (à supprimer plus tard quand on utilisera main.py)
if __name__ == "__main__":
    # Remplace par le bon chemin vers ton fichier test
    # load_and_show_dicom("../data/mon_image_test.dcm")
    pass