pour actualiser les biblio:
python -m pip freeze > requirements.txt

toujours dans le .venv

pour les image deja traiter par le medecin: /03_processed
    pour applique le masque ia 
    pour visualiser des imagerie deja completé + avec l'ia local : python predict_and_visualize.py
    pour voir le pancrea localement : python view_extracted_cube.py


pour les images non traité : /02_interim
    pour visualiser des imagerie : python visualize_raw_prediction.py
    pour voir le pancrea localement : python view_predicted_cube.py

    pour voir la prediction de survie : python  predict_survival_pipeline.py
