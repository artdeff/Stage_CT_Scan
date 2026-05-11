% =========================================================================
% SCRIPT : export_to_hdf5.m
% RÔLE   : Charger une série DICOM, convertir en HU et exporter en HDF5
% =========================================================================

clear; clc; close all;

% --- 0. Robustesse : Détection de l'environnement (MATLAB ou Octave) ---
% Si la variable système 'OCTAVE_VERSION' existe, nous sommes sous Octave.
if exist('OCTAVE_VERSION', 'builtin') ~= 0
    disp('🔵 Environnement Octave détecté. Chargement des paquets...');
    pkg load dicom;  % Indispensable pour dicominfo et dicomread
    % pkg load image; % Décommente ceci plus tard si tu utilises des filtres
else
    disp('🔴 Environnement MATLAB détecté.');
end

% --- 1. Paramètres des chemins ---
input_dir = '..\data\01_raw\patient_01'; % Dossier contenant les .dcm


% --- 1. Paramètres des chemins ---
% Utilisation de chemins relatifs par rapport au dossier matlab/
input_dir = '..\data\01_raw\patient_01'; % Dossier contenant les .dcm
output_file = '..\data\02_interim\patient_01.h5';
dataset_name = '/volume_ct'; % Nom de l'arborescence interne du HDF5

% --- 2. Lecture des métadonnées (via la première coupe) ---
files = dir(fullfile(input_dir, '*.dcm'));
if isempty(files)
    error('⚠️ Aucun fichier DICOM trouvé dans le dossier spécifié.');
end

% On extrait les infos du premier fichier pour les métadonnées
info = dicominfo(fullfile(input_dir, files(1).name));

% --- EXTRACTION ROBUSTE DES METADONNÉES ---

% 1. Espacement des pixels (X, Y)
if isfield(info, 'PixelSpacing')
    pixel_spacing = info.PixelSpacing;
else
    warning('⚠️ PixelSpacing introuvable. Valeur par défaut [1; 1] appliquée. Attention pour la Radiomics !');
    pixel_spacing = [1; 1];
end

% 2. Épaisseur de la coupe (Z)
if isfield(info, 'SliceThickness')
    slice_thickness = info.SliceThickness;
else
    warning('⚠️ SliceThickness introuvable. Valeur par défaut 1 appliquée.');
    slice_thickness = 1;
end

% 3. Conversion Hounsfield (Pente)
if isfield(info, 'RescaleSlope')
    rescale_slope = info.RescaleSlope;
else
    rescale_slope = 1; % Par défaut, pas de mise à l'échelle
end

% 4. Conversion Hounsfield (Ordonnée à l'origine)
if isfield(info, 'RescaleIntercept')
    rescale_intercept = info.RescaleIntercept;
else
    rescale_intercept = 0; % Par défaut, pas de décalage
end

% 5. ID Patient
if isfield(info, 'PatientID')
    patient_id = info.PatientID;
else
    patient_id = 'Inconnu';
end

fprintf('Patient ID: %s\n', patient_id);
% Attention: Octave peut utiliser 'Rows' ou 'Height', 'Columns' ou 'Width'
if isfield(info, 'Rows')
    H = info.Rows; W = info.Columns;
else
    H = info.Height; W = info.Width; % Alternative pour certaines images
end
fprintf('Dimensions de la coupe: %d x %d\n', H, W);

% --- 3. Lecture du volume 3D ---
D = length(files);
H = info.Rows;
W = info.Columns;

% Bonne pratique : Pré-allocation de la mémoire pour la vitesse
volume_raw = zeros(H, W, D, 'int16'); 

disp('⏳ Chargement du volume DICOM...');
for z = 1:D
    % Lecture de chaque coupe et insertion dans le tenseur 3D
    volume_raw(:, :, z) = dicomread(fullfile(input_dir, files(z).name));
end

% --- 4. Conversion en Unités Hounsfield (HU) ---
disp('⚙️ Conversion en unités Hounsfield...');
volume_hu = double(volume_raw) * rescale_slope + rescale_intercept;

% --- 5. Exportation au format HDF5 ---
disp('💾 Création du fichier HDF5...');

% Sécurité : supprimer le fichier s'il existe déjà pour éviter les conflits
if isfile(output_file)
    delete(output_file);
end

% Étape A : Créer l'espace dans le fichier (Datatype 'double' pour les HU)
h5create(output_file, dataset_name, size(volume_hu), 'Datatype', 'double');

% Étape B : Écrire la matrice 3D
h5write(output_file, dataset_name, volume_hu);

% Étape C : Attacher les métadonnées vitales comme "Attributs"
h5writeatt(output_file, dataset_name, 'PixelSpacing', pixel_spacing);
h5writeatt(output_file, dataset_name, 'SliceThickness', slice_thickness);
h5writeatt(output_file, dataset_name, 'PatientID', info.PatientID);

disp('✅ Export HDF5 terminé avec succès !');