# src/config.py

from pathlib import Path

# --- 1. Configuração de Caminhos ---

# Raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Diretório base contendo os datasets
DATASET_DIR = PROJECT_ROOT / 'dataset'

# Diretórios de dados originais
ORIGINAL_TRAINING_DIR = DATASET_DIR / 'Training'
ORIGINAL_TESTING_DIR = DATASET_DIR / 'Testing'

# Diretórios para salvar os dados pré-processados
CLEANED_TRAINING_DIR = DATASET_DIR / 'cleaned' / 'Training'
CLEANED_TESTING_DIR = DATASET_DIR / 'cleaned' / 'Testing'


# Diretório para salvar o modelo treinado
MODEL_DIR = PROJECT_ROOT / 'model'
BEST_MODEL_PATH = MODEL_DIR / 'brain_tumor_best_model.h5'

# Diretório para salvar gráficos e resultados
PLOTS_DIR = MODEL_DIR / 'plots'

# --- 2. Parâmetros do Modelo e Treinamento ---
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.2

# --- 3. Garantir que os diretórios de saída existam ---
CLEANED_TRAINING_DIR.mkdir(parents=True, exist_ok=True)
CLEANED_TESTING_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)