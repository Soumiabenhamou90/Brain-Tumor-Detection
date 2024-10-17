from brainTumorDetection.ml_logic.preprocessingData.splitData import custom_train_test_split
dataset_BrainTumorLabeled = "raw_data/BrainTumorlabeleddataset/"

def make_preprocessing():
    print("start split data (preprocessing ...)")
    custom_train_test_split(dataset_BrainTumorLabeled)
    return True


make_preprocessing()
