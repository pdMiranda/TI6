import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Model

def plot_training_performance(history, output_dir: Path):
    """
    Plota metricas de treinamento
    """
    # Grafico de Acuracia
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['accuracy'], 'r', linewidth=3.0)
    plt.plot(history.history['val_accuracy'], 'b', linewidth=3.0)
    plt.legend(['Acuracia de Treinamento', 'Acuracia de Validacao'], fontsize=18)
    plt.xlabel('Epocas', fontsize=16)
    plt.ylabel('Acuracia', fontsize=16)
    plt.title('Curvas de Acuracia', fontsize=16)
    plt.savefig(output_dir / 'training_accuracy.png')
    plt.close()

    # Grafico de Perda
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Perda de Treinamento', 'Perda de Validacao'], fontsize=18)
    plt.xlabel('Epocas', fontsize=16)
    plt.ylabel('Perda', fontsize=16)
    plt.title('Curvas de Perda', fontsize=16)
    plt.savefig(output_dir / 'training_loss.png')
    plt.close()

def evaluate_model(model: Model, test_dataset, output_dir: Path, class_names: list):
    """
    Avalia o modelo no conjunto de teste
    """
    print("Avaliando modelo no conjunto de teste...")
    
    # Avaliacao basica
    loss, acc = model.evaluate(test_dataset, verbose=0)
    print(f"Acuracia no conjunto de teste: {acc*100:.2f}%")
    print(f"Loss no conjunto de teste: {loss:.4f}")

    # Coleta previsoes e labels verdadeiros
    print("Gerando previsoes...")
    y_pred = []
    y_true = []
    
    for images, labels in test_dataset:
        # Predicoes do modelo
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        
        # Labels verdadeiros - converte one-hot para indices
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    # Garante que sao arrays numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    print(f"Shape y_true: {y_true.shape}, y_pred: {y_pred.shape}")

    # Relatorio de classificacao
    print("\nRelatorio de Classificacao:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Matriz de confusao
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusao')
    plt.xlabel('Previsto')
    plt.ylabel('Verdadeiro')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Resultados salvos em: {output_dir}")