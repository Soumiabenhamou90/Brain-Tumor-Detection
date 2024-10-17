import os
import shutil
import random
from brainTumorDetection.ml_logic.preprocessingData.removeHiddenFolders import remove_hidden_folders
from sklearn.model_selection import train_test_split
def custom_train_test_split(dataset_directory, test_ratio=0.2, random_seed=None, min_samples_per_class=2):
    """
    Custom function to perform a stratified train-test split on an image dataset.
    This function organizes the dataset into 'train' and 'test' directories while maintaining the class-wise distribution.
    It creates the 'train' and 'test' directories within the given dataset directory and moves images accordingly.

    """
    remove_hidden_folders(dataset_directory)#j'appel la fonction pour supprimer les dossier avec un "."

    random_seed = random_seed or 42

    class_directories = [d for d in os.listdir(dataset_directory) if os.path.isdir(os.path.join(dataset_directory, d))]


    train_directory = os.path.join(dataset_directory, 'train')
    test_directory = os.path.join(dataset_directory, 'test')
    os.makedirs(train_directory, exist_ok=True)
    os.makedirs(test_directory, exist_ok=True)


    for class_dir in class_directories:

        class_path = os.path.join(dataset_directory, class_dir, "images/")
        class_path_labels = os.path.join(dataset_directory, class_dir, "labels/")

        image_files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]


        if len(image_files) < min_samples_per_class:
            continue

        #split des donnees en train et test dateset
        train_images, test_images = train_test_split(image_files, test_size=test_ratio, random_state=random_seed)


        for train_image in train_images:
            src_image_path = os.path.join(class_path, train_image)
            src_annotation_path = os.path.join(class_path_labels, train_image.replace('.jpg', '.txt'))

            dest_image_path = os.path.join(train_directory, class_dir, "images/")
            dest_annotation_path = os.path.join(train_directory, class_dir, "labels/")

            os.makedirs(os.path.dirname(dest_image_path), exist_ok=True)
            os.makedirs(os.path.dirname(dest_annotation_path), exist_ok=True)

            if os.path.exists(src_annotation_path):
                shutil.copy(src_image_path, dest_image_path)
                shutil.copy(src_annotation_path, dest_annotation_path)

        for test_image in test_images:
            src_image_path = os.path.join(class_path, test_image)
            src_annotation_path = os.path.join(class_path_labels, test_image.replace('.jpg', '.txt'))

            dest_image_path = os.path.join(test_directory, class_dir, "images/")
            dest_annotation_path = os.path.join(test_directory, class_dir, "labels/")

            print(src_image_path, dest_annotation_path)

            os.makedirs(os.path.dirname(dest_image_path), exist_ok=True)
            os.makedirs(os.path.dirname(dest_annotation_path), exist_ok=True)

            if os.path.exists(src_annotation_path):
                shutil.copy(src_image_path, dest_image_path)
                shutil.copy(src_annotation_path, dest_annotation_path)
    return True
