"""
=============================================================================
MODULE : alternative_survival_model.py
RÔLE   : CNN 3D Léger ("Le Petit Poucet") pour éviter le Mode Collapse.
         Remplace l'architecture trop lourde d'EfficientNet.
=============================================================================
"""

import torch
import torch.nn as nn

class SimpleCNN3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super(SimpleCNN3D, self).__init__()
        
        # Bloc 1 : Extraction des premiers contours (Bords, textures simples)
        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(2) # Réduit la taille du cube par 2
        
        # Bloc 2 : Motifs moyens
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(2)
        
        # Bloc 3 : Motifs complexes et denses
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool3d(2)
        
        # Pooling global adaptatif pour s'assurer que le modèle accepte 
        # des cubes de tailles légèrement différentes sans planter
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Couche de décision finale (Sortie vers nos 3 classes)
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1) # Aplatit les données en 1D
        x = self.fc(x)
        return x

def create_alternative_survival_classifier(spatial_dims=3, in_channels=1, num_classes=3):
    """
    On garde le même nom de fonction pour que vos scripts d'entraînement 
    fonctionnent sans aucune modification !
    """
    return SimpleCNN3D(in_channels=in_channels, num_classes=num_classes)