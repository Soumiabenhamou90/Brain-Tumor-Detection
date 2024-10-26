
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Configuration de l'ImageDataGenerator
datagen = ImageDataGenerator(
     rescale=1./255,  # Normaliser les pixels entre 0 et 1
    rotation_range=20,  # Rotation aléatoire entre -20 et +20 degrés
    width_shift_range=0.2,  # Décalage horizontal jusqu'à 20%
    height_shift_range=0.2,  # Décalage vertical jusqu'à 20%
    shear_range=0.2,  # Cisaillement aléatoire jusqu'à 20%
    zoom_range=0.2,  # Zoom aléatoire jusqu'à 20%
    horizontal_flip=True,  # Retourner horizontalement
    fill_mode='nearest',  # Remplissage avec les pixels les plus proches
    validation_split=0.2  # 20% des données pour le test
)
# Normalisation et split des données
train_generator = datagen.flow_from_directory('/content/drive/MyDrive/BrainTumorLabeledDataset/train',
target_size=(224, 224), # Taille pour VGG16
batch_size=32,
class_mode='categorical', # Classification multi-classes
subset='training' )

validation_generator = datagen.flow_from_directory(
'/content/drive/MyDrive/BrainTumorLabeledDataset/train',
target_size=(224, 224),
batch_size=32,
class_mode='categorical',
subset='validation' ) # Charger VGG16 préentraîné sur ImageNet

# Générateur pour le test set
test_generator = datagen.flow_from_directory(
    '/content/drive/MyDrive/BrainTumorLabeledDataset/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)






from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# Chargement du modèle ResNet50 préentraîné sur ImageNet

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Geler les poids du modèle de base
base_model.trainable = False

# Ajouter des couches au-dessus de ResNet50

x = base_model.output
x = GlobalAveragePooling2D()(x)  # Pooling global
x = Dense(1024, activation='relu')(x)  # Couche fully connected
x = Dropout(0.5)(x)  # Ajout de Dropout pour éviter le sur-apprentissage
predictions = Dense(4, activation='softmax')(x)  # 4 classes pour la sortie

# Définir le modèle final
model = Model(inputs=base_model.input, outputs=predictions)

# Compiler le modèle
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['recall'])

# Résumé du modèle
model.summary()






# Entraînement du modèle
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=100,  # Vous pouvez ajuster ce nombre selon vos besoins
    verbose=1
)

# Évaluation du modèle sur les données de test
test_loss, test_recall = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Test Loss: {test_loss}, Test Recall: {test_recall}')
