import os
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Importações do TensorFlow e Scikit-learn
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# Garante a reprodutibilidade dos resultados
np.random.seed(42)
tf.random.set_seed(42)

def crop_img(img: np.ndarray) -> np.ndarray:
    """
    Função de pré-processamento para cortar a imagem e focar na região do cérebro.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if not cnts:
        return img  # Retorna a imagem original se nenhum contorno for encontrado

    c = max(cnts, key=cv2.contourArea)

    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    new_img = img[extTop[1]:extBot[1], extLeft[0]:extRight[0]].copy()
    return new_img

def preprocess_and_save_dataset(source_dir: Path, dest_dir: Path, img_size: int):
    """
    Aplica o pré-processamento (corte e redimensionamento) nas imagens de um diretório
    e salva o resultado em um novo diretório, evitando trabalho redundante.
    """
    print(f"Iniciando pré-processamento de '{source_dir}' para '{dest_dir}'...")
    labels = [d.name for d in source_dir.iterdir() if d.is_dir()]
    
    for label in labels:
        source_label_dir = source_dir / label
        dest_label_dir = dest_dir / label
        dest_label_dir.mkdir(parents=True, exist_ok=True)

        for img_file in source_label_dir.glob('*'):
            try:
                image = cv2.imread(str(img_file))
                if image is None:
                    print(f"Aviso: Não foi possível ler a imagem {img_file}")
                    continue
                
                cropped_image = crop_img(image)
                resized_image = cv2.resize(cropped_image, (img_size, img_size))
                
                save_path = dest_label_dir / img_file.name
                cv2.imwrite(str(save_path), resized_image)
            except Exception as e:
                print(f"Erro ao processar {img_file}: {e}")
                
    print("Pré-processamento e salvamento concluídos.")


def create_model(image_size: int, num_classes: int) -> Model:
    """
    Cria o modelo CNN usando Transfer Learning com a ResNet50.
    """
    base_net = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(image_size, image_size, 3)
    )
    base_net.trainable = False

    model = base_net.output
    model = GlobalAveragePooling2D()(model)
    model = Dropout(0.4)(model)
    model = Dense(num_classes, activation="softmax")(model)
    
    final_model = Model(inputs=base_net.input, outputs=model)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    final_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return final_model

# A função agora recebe o diretório de saída como argumento
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

def main():
    """Função principal para executar o fluxo de trabalho."""
    # --- 1. Configuração de Caminhos e Parâmetros com pathlib ---
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    BASE_DIR = PROJECT_ROOT / 'dataset'
    
    ORIGINAL_TRAINING_DIR = BASE_DIR / 'Training'
    ORIGINAL_TESTING_DIR = BASE_DIR / 'Testing'
    
    CLEANED_TRAINING_DIR = BASE_DIR / 'cleaned' / 'Training'
    CLEANED_TESTING_DIR = BASE_DIR / 'cleaned' / 'Testing'
    
    OUTPUT_DIR = BASE_DIR / 'plots'
    # Garante que o diretório de saída exista
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 15

    # --- 2. Pré-processamento (Executado apenas uma vez) ---
    if not CLEANED_TRAINING_DIR.exists() or not any(CLEANED_TRAINING_DIR.iterdir()):
        preprocess_and_save_dataset(ORIGINAL_TRAINING_DIR, CLEANED_TRAINING_DIR, IMG_SIZE)
    else:
        print("Diretório de treinamento pré-processado já existe. Pulando etapa.")

    if not CLEANED_TESTING_DIR.exists() or not any(CLEANED_TESTING_DIR.iterdir()):
        preprocess_and_save_dataset(ORIGINAL_TESTING_DIR, CLEANED_TESTING_DIR, IMG_SIZE)
    else:
        print("Diretório de teste pré-processado já existe. Pulando etapa.")

    # --- 3. Geração de Dados (Eficiente em Memória) ---
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        CLEANED_TRAINING_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        CLEANED_TRAINING_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    test_generator = test_datagen.flow_from_directory(
        CLEANED_TESTING_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # --- 4. Construção e Treinamento do Modelo ---
    print("\nConstruindo o modelo...")
    num_classes = len(train_generator.class_indices)
    model = create_model(IMG_SIZE, num_classes)
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1, mode='min'),
        ModelCheckpoint(filepath='brain_tumor_best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    print("\nIniciando o treinamento...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=callbacks
    )

    # --- 5. Avaliação do Modelo ---
    print("\nAvaliando o modelo treinado...")
    # Passa o diretório de saída para a função de plotagem
    plot_and_save_history(history, OUTPUT_DIR, "training_history")
    print(f"Gráficos de treinamento salvos em: {OUTPUT_DIR}")

    loss, acc = model.evaluate(test_generator, verbose=0)
    print(f"\nAcurácia no conjunto de teste: {acc*100:.2f}%")

    predicted_classes_prob = model.predict(test_generator)
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
    # Salva a matriz de confusão no diretório de saída
    plt.savefig(OUTPUT_DIR / 'confusion_matrix.png')
    print(f"Matriz de confusão salva em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()