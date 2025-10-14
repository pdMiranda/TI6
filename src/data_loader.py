import tensorflow as tf
import os
import glob

def create_data_pipeline(data_dir, batch_size, img_size, is_training=False):
    """
    Pipeline de dados simples e eficiente
    """
    # Encontra todas as imagens
    image_patterns = [
        os.path.join(data_dir, "*", "*.jpg"),
        os.path.join(data_dir, "*", "*.png"), 
        os.path.join(data_dir, "*", "*.jpeg")
    ]
    
    image_paths = []
    for pattern in image_patterns:
        image_paths.extend(glob.glob(pattern))
    
    if not image_paths:
        raise ValueError(f"Nenhuma imagem encontrada em {data_dir}")
    
    # Extrai labels dos paths
    labels = [os.path.basename(os.path.dirname(path)) for path in image_paths]
    class_names = sorted(set(labels))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    labels_idx = [class_to_idx[label] for label in labels]
    
    # Cria dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels_idx))
    
    # Shuffle apenas para treino
    if is_training:
        dataset = dataset.shuffle(len(image_paths))
    
    # Funcao para carregar e pre-processar
    def load_and_preprocess_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, [img_size, img_size])
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.one_hot(label, len(class_names))
        return image, label
    
    # Augmentations simples para treino
    def augment_image(image, label):
        if is_training:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.1)
        return image, label
    
    # Aplica as transformacoes
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch e prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, class_names