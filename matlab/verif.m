pkg load dicom;
info = dicominfo('..\data\01_raw\patient_01\testtttt.dcm');

% On regarde dans SharedFunctionalGroupsSequence
if isfield(info, 'SharedFunctionalGroupsSequence')
    disp('Contenu de SharedFunctionalGroupsSequence.Item_1 :');
    disp(fieldnames(info.SharedFunctionalGroupsSequence.Item_1));
    
    % Test de chemins alternatifs courants en IRM
    try
        % Parfois c'est directement dans Item_1
        sp = info.SharedFunctionalGroupsSequence.Item_1.PixelSpacing;
        disp(['Trouvé ! PixelSpacing = ', num2str(sp')]);
    catch
        disp('Toujours pas de PixelSpacing direct...');
    end
end

% On vérifie aussi le premier élément de la séquence par cadre
if isfield(info, 'PerFrameFunctionalGroupsSequence')
     disp('Contenu de PerFrameFunctionalGroupsSequence.Item_1 :');
     disp(fieldnames(info.PerFrameFunctionalGroupsSequence.Item_1));
end