import tensorflow as tf
import numpy as np

# Importa as configurações e módulos do projeto
import config
from preprocessing import process_and_save_dataset
from model_architecture import create_model
from training import get_data_generators, train_model
from evaluation import plot_and_save_history, evaluate_model

# Garante a reprodutibilidade
np.random.seed(42)
tf.random.set_seed(42)

def main():
    """Função principal para executar o fluxo de trabalho completo."""
    
    # --- 1. Pré-processamento (Executado apenas se necessário) ---
    if not any(config.CLEANED_TRAINING_DIR.iterdir()):
        process_and_save_dataset(config.ORIGINAL_TRAINING_DIR, config.CLEANED_TRAINING_DIR, config.IMG_SIZE)
    else:
        print("Diretório de treinamento pré-processado já existe. Pulando etapa.")

    if not any(config.CLEANED_TESTING_DIR.iterdir()):
        process_and_save_dataset(config.ORIGINAL_TESTING_DIR, config.CLEANED_TESTING_DIR, config.IMG_SIZE)
    else:
        print("Diretório de teste pré-processado já existe. Pulando etapa.")

    # --- 2. Preparação dos Dados ---
    print("\nCarregando geradores de dados...")
    train_gen, val_gen, test_gen = get_data_generators()

    # --- 3. Construção do Modelo ---
    print("\nConstruindo o modelo...")
    num_classes = len(train_gen.class_indices)
    model = create_model(config.IMG_SIZE, num_classes, config.LEARNING_RATE)
    model.summary()

    # --- 4. Treinamento do Modelo ---
    history = train_model(model, train_gen, val_gen)

    # --- 5. Avaliação do Modelo ---
    print("\nAvaliando o modelo treinado...")
    
    # Salva os gráficos de acurácia e perda
    plot_and_save_history(history, config.PLOTS_DIR, "training_history")
    print(f"Gráficos de treinamento salvos em: {config.PLOTS_DIR}")

    # Avalia no conjunto de teste e salva os resultados
    evaluate_model(model, test_gen, config.PLOTS_DIR)


if __name__ == "__main__":
    main()