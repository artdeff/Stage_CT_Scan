clear; clc; close all;

if exist('OCTAVE_VERSION', 'builtin') ~= 0
    pkg load dicom;
end

% --- 1. Paramètres ---
input_file = '..\data\01_raw\patient_01\testtttt.dcm'; % Un fichier DICOM multi-frame d'IRM
output_file = '..\data\02_interim\patient_01.h5';
dataset_name = '/volume_mri';

% --- 2. Lecture ---
info = dicominfo(input_file);
% dicomread sur un multi-frame renvoie souvent un tenseur 4D (H x W x 1 x N)
volume_raw = dicomread(input_file);

% On squeeze pour passer de (H, W, 1, 21) à (H, W, 21)
volume_mri = squeeze(volume_raw);

% --- 3. Métadonnées (Valeurs "Fallback" si on ne trouve pas) ---
% On utilisera les valeurs trouvées avec verif.m ici
pixel_spacing = [1.0; 1.0]; 
slice_thickness = 1.0;

% --- 4. Export HDF5 ---
if isfile(output_file); delete(output_file); end

h5create(output_file, dataset_name, size(volume_mri), 'Datatype', 'single');
h5write(output_file, dataset_name, single(volume_mri)); % 'single' = float32 en Python

h5writeatt(output_file, dataset_name, 'PixelSpacing', pixel_spacing);
h5writeatt(output_file, dataset_name, 'SliceThickness', slice_thickness);
h5writeatt(output_file, dataset_name, 'Modality', 'MR');

disp('✅ Volume IRM exporté en HDF5 !');