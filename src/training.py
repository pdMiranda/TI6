import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
import config  # Importação direta

def get_data_generators():
    """Cria e retorna os geradores de dados para treino, validação e teste."""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        validation_split=config.VALIDATION_SPLIT
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        config.CLEANED_TRAINING_DIR,
        target_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        config.CLEANED_TRAINING_DIR,
        target_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    test_generator = test_datagen.flow_from_directory(
        config.CLEANED_TESTING_DIR,
        target_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator

def train_model(model: Model, train_gen, val_gen):
    """Executa o treinamento do modelo."""
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1, mode='min'),
        ModelCheckpoint(filepath=str(config.BEST_MODEL_PATH), monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    print("\nIniciando o treinamento do modelo...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        steps_per_epoch=train_gen.samples // config.BATCH_SIZE,
        validation_steps=val_gen.samples // config.BATCH_SIZE,
        callbacks=callbacks
    )
    return history