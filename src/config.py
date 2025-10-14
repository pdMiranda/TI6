from pathlib import Path

# --- 1. Configuração de Caminhos ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / 'dataset'

ORIGINAL_TRAINING_DIR = DATASET_DIR / 'Training'
ORIGINAL_TESTING_DIR = DATASET_DIR / 'Testing'

CLEANED_TRAINING_DIR = DATASET_DIR / 'cleaned' / 'Training'
CLEANED_TESTING_DIR = DATASET_DIR / 'cleaned' / 'Testing'

MODEL_DIR = PROJECT_ROOT / 'model'
BEST_MODEL_PATH = MODEL_DIR / 'brain_tumor_best_model.keras'
PLOTS_DIR = MODEL_DIR / 'plots'

# --- 2. Parâmetros Otimizados para GPU ---
IMG_SIZE = 224
BATCH_SIZE = 64  
EPOCHS = 50      
LEARNING_RATE = 0.1
VALIDATION_SPLIT = 0.2

# --- 3. Garantir que os diretórios de saída existam ---
CLEANED_TRAINING_DIR.mkdir(parents=True, exist_ok=True)
CLEANED_TESTING_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
(MODEL_DIR / 'logs').mkdir(parents=True, exist_ok=True)