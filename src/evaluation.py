import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import DirectoryIterator

def plot_and_save_history(history, output_dir: Path, file_prefix: str):
    """
    Plota e salva as curvas de acurácia e perda no diretório especificado.
    """
    # Gráfico de Acurácia
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['accuracy'], 'r', linewidth=3.0)
    plt.plot(history.history['val_accuracy'], 'b', linewidth=3.0)
    plt.legend(['Acurácia de Treinamento', 'Acurácia de Validação'], fontsize=18)
    plt.xlabel('Épocas', fontsize=16)
    plt.ylabel('Acurácia', fontsize=16)
    plt.title('Curvas de Acurácia', fontsize=16)
    plt.savefig(output_dir / f'{file_prefix}_accuracy.png')
    plt.close()

    # Gráfico de Perda
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Perda de Treinamento', 'Perda de Validação'], fontsize=18)
    plt.xlabel('Épocas', fontsize=16)
    plt.ylabel('Perda', fontsize=16)
    plt.title('Curvas de Perda', fontsize=16)
    plt.savefig(output_dir / f'{file_prefix}_loss.png')
    plt.close()

def evaluate_model(model: Model, test_generator: DirectoryIterator, output_dir: Path):
    """
    Avalia o modelo no conjunto de teste e salva o relatório e a matriz de confusão.
    """
    loss, acc = model.evaluate(test_generator, verbose=0)
    print(f"\nAcurácia no conjunto de teste: {acc*100:.2f}%")

    predicted_classes_prob = model.predict(test_generator, verbose=0)
    predicted_classes = np.argmax(predicted_classes_prob, axis=1)
    true_classes = test_generator.classes
    class_names = list(test_generator.class_indices.keys())

    print("\nRelatório de Classificação:\n")
    print(classification_report(true_classes, predicted_classes, target_names=class_names))

    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap='Blues', annot=True, cbar=True, fmt='d', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusão')
    plt.xlabel('Previsto')
    plt.ylabel('Verdadeiro')
    plt.savefig(output_dir / 'confusion_matrix.png')
    plt.close()
    
    print(f"Relatório e matriz de confusão salvos em: {output_dir}")