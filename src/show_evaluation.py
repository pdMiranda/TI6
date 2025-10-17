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
from data_loader import create_data_pipeline
from evaluation import evaluate_model

# Reprodutibilidade
np.random.seed(42)
tf.random.set_seed(42)

def main():
    """Fluxo principal para avaliação do modelo existente"""

    # Carrega o modelo existente
    print("Carregando modelo existente...")
    model_path = config.BEST_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {model_path}")

    model = tf.keras.models.load_model(model_path)
    print("Modelo carregado com sucesso.")

    # Carrega os dados de teste
    print("Carregando dados de teste...")
    test_dataset, class_names = create_data_pipeline(
        str(config.CLEANED_TESTING_DIR),
        config.BATCH_SIZE,
        config.IMG_SIZE,
        is_training=False
    )

    print("Classes detectadas:", class_names)

    # Avalia o modelo
    print("Avaliando modelo...")
    evaluate_model(model, test_dataset, config.PLOTS_DIR, class_names)

if __name__ == "__main__":
    main()