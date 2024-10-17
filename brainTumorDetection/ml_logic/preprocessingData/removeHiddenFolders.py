#supprimer les dossier qui commence par un .
import os
import shutil
import random
def remove_hidden_folders(root_dir):
    # Parcours du répertoire racine et des sous-répertoires
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            # Vérifier si le nom du dossier commence par un point
            if dir_name.startswith('.'):
                dir_path = os.path.join(root, dir_name)
                print(f"Suppression du dossier: {dir_path}")
                shutil.rmtree(dir_path)  # Supprimer le dossier et tout son contenu
    return True
