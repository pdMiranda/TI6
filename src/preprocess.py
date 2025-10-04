import os
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Para garantir a reprodutibilidade dos resultados
np.random.seed(42)
tf.random.set_seed(42)

def crop_img(img):
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
        return img 
    c = max(cnts, key=cv2.contourArea)

    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    new_img = img[extTop[1]:extBot[1], extLeft[0]:extRight[0]].copy()
    
    return new_img

def load_and_preprocess_data(training_path, testing_path, image_size):
    """
    Carrega os dados, aplica o pré-processamento de corte e redimensionamento.
    Usa OpenCV (cv2) para manipulação de imagem e NumPy para manipulação de arrays.
    """
    labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    x_train, y_train, x_test, y_test = [], [], [], []

    print("Carregando e pré-processando dados de treinamento...")
    for label in labels:
        train_dir = os.path.join(training_path, label)
        for file in os.listdir(train_dir):
            image_path = os.path.join(train_dir, file)
            image = cv2.imread(image_path)
            image = crop_img(image)
            image = cv2.resize(image, (image_size, image_size))
            x_train.append(image)
            y_train.append(labels.index(label))

    print("Carregando e pré-processando dados de teste...")
    for label in labels:
        test_dir = os.path.join(testing_path, label)
        for file in os.listdir(test_dir):
            image_path = os.path.join(test_dir, file)
            image = cv2.imread(image_path)
            image = crop_img(image)
            image = cv2.resize(image, (image_size, image_size))
            x_test.append(image)
            y_test.append(labels.index(label))

    x_train = np.array(x_train) / 255.0
    x_test = np.array(x_test) / 255.0
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test, labels

def create_model(image_size, num_classes):
    """
    Cria o modelo CNN usando Transfer Learning com Keras.
    """
    # Carrega o modelo ResNet50 pré-treinado (disponibilizado pelo Keras)
    base_net = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(image_size, image_size, 3)
    )

    # Adiciona camadas customizadas no topo da ResNet50 (funcionalidade do Keras)
    model = base_net.output
    model = GlobalAveragePooling2D()(model)
    model = Dropout(0.4)(model)
    model = Dense(num_classes, activation="softmax")(model)
    model = Model(inputs=base_net.input, outputs=model)

    # Compila o modelo, definindo otimizador e função de perda (funcionalidade do Keras)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def plot_history(history, file_prefix):
    """
    Plota e salva as curvas de acurácia e perda usando Matplotlib.
    """
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Perda de Treinamento', 'Perda de Validação'], fontsize=18)
    plt.xlabel('Épocas', fontsize=16)
    plt.ylabel('Perda', fontsize=16)
    plt.title('Curvas de Perda', fontsize=16)
    plt.savefig(f'{file_prefix}_loss_curves.png')
    plt.close()

    plt.figure(figsize=[8, 6])
    plt.plot(history.history['accuracy'], 'r', linewidth=3.0)
    plt.plot(history.history['val_accuracy'], 'b', linewidth=3.0)
    plt.legend(['Acurácia de Treinamento', 'Acurácia de Validação'], fontsize=18)
    plt.xlabel('Épocas', fontsize=16)
    plt.ylabel('Acurácia', fontsize=16)
    plt.title('Curvas de Acurácia', fontsize=16)
    plt.savefig(f'{file_prefix}_accuracy_curves.png')
    plt.close()

if __name__ == "__main__":
    BASE_DIR = './' 
    TRAINING_DIR = os.path.join(BASE_DIR, 'dataset/Training')
    TESTING_DIR = os.path.join(BASE_DIR, 'dataset/Testing')
    IMAGE_SIZE = 200
    BATCH_SIZE = 32
    EPOCHS = 15

    # Carrega e prepara os dados
    x_train, y_train, x_test, y_test, class_names = load_and_preprocess_data(
        TRAINING_DIR, TESTING_DIR, IMAGE_SIZE
    )

    # Embaralha e faz o One-Hot Encoding dos rótulos
    x_train, y_train = shuffle(x_train, y_train, random_state=42)
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    # Divide os dados em treino e validação usando scikit-learn
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )

    # Prepara o gerador de imagens para Data Augmentation (Keras)
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True
    )
    datagen.fit(x_train)

    # Constrói o modelo CNN
    print("\nConstruindo o modelo...")
    model = create_model(IMAGE_SIZE, num_classes=len(class_names))
    model.summary()
    
    # Treina o modelo (Keras)
    print("\nIniciando o treinamento...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1, mode='min'),
        ModelCheckpoint(filepath='brain_tumor_best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
    ]

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Avalia o modelo
    print("\nAvaliando o modelo treinado...")
    
    plot_history(history, "training_history")
    print("Gráficos de treinamento salvos.")

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nAcurácia no conjunto de teste: {acc*100:.2f}%")

    predicted_classes = np.argmax(model.predict(x_test), axis=1)
    true_classes = np.argmax(y_test, axis=1)

    # Gera relatório e matriz de confusão (scikit-learn, matplotlib, seaborn)
    print("\nRelatório de Classificação:\n")
    print(classification_report(true_classes, predicted_classes, target_names=class_names))

    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap='Blues', annot=True, cbar=True, fmt='d', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusão')
    plt.xlabel('Previsto')
    plt.ylabel('Verdadeiro')
    plt.savefig('confusion_matrix.png')
    print("Matriz de confusão salva.")