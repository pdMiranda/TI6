import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# Configuracao da GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU configurada:", gpus[0].name)
    except RuntimeError as e:
        print("GPU ja inicializada:", e)

import numpy as np

import config
from preprocessing import process_and_save_dataset
from model_architecture import create_optimized_model
from data_loader import create_data_pipeline
from training import train_model
from evaluation import plot_training_performance, evaluate_model

# Reprodutibilidade
np.random.seed(42)
tf.random.set_seed(42)

def main():
    """Fluxo principal"""
    
    # 1. Pre-processamento (se necessario)
    if not any(config.CLEANED_TRAINING_DIR.iterdir()):
        print("Pre-processando dados de treinamento...")
        process_and_save_dataset(config.ORIGINAL_TRAINING_DIR, config.CLEANED_TRAINING_DIR, config.IMG_SIZE)
    
    if not any(config.CLEANED_TESTING_DIR.iterdir()):
        print("Pre-processando dados de teste...")
        process_and_save_dataset(config.ORIGINAL_TESTING_DIR, config.CLEANED_TESTING_DIR, config.IMG_SIZE)
    
    # 2. Carrega dados
    print("Carregando datasets...")
    train_dataset, class_names = create_data_pipeline(
        str(config.CLEANED_TRAINING_DIR),
        config.BATCH_SIZE,
        config.IMG_SIZE,
        is_training=True
    )
    
    val_dataset, _ = create_data_pipeline(
        str(config.CLEANED_TRAINING_DIR),
        config.BATCH_SIZE,
        config.IMG_SIZE,
        is_training=False
    )
    
    test_dataset, _ = create_data_pipeline(
        str(config.CLEANED_TESTING_DIR),
        config.BATCH_SIZE,
        config.IMG_SIZE,
        is_training=False
    )
    
    print("Classes detectadas:", class_names)
    
    # 3. Cria modelo
    print("Construindo modelo...")
    model = create_optimized_model(config.IMG_SIZE, len(class_names), config.LEARNING_RATE)
    model.summary()
    
    # 4. Treina
    history = train_model(model, train_dataset, val_dataset)
    
    # 5. Avalia
    print("Avaliando modelo...")
    plot_training_performance(history, config.PLOTS_DIR)
    evaluate_model(model, test_dataset, config.PLOTS_DIR, class_names)

if __name__ == "__main__":
    main()