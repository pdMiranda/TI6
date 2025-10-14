import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import config

def get_callbacks():
    """
    Callbacks essenciais
    """
    return [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(config.BEST_MODEL_PATH),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

def train_model(model, train_dataset, val_dataset):
    """
    Treinamento direto
    """
    callbacks = get_callbacks()
    
    print("Iniciando treinamento")
    print("Batch size:", config.BATCH_SIZE)
    print("Epocas:", config.EPOCHS)
    
    # Treinamento
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    return history