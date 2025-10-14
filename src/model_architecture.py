import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam

def create_optimized_model(image_size: int, num_classes: int, learning_rate: float) -> Model:
    """
    Modelo direto e eficiente
    """
    base_net = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(image_size, image_size, 3)
    )
    base_net.trainable = False

    x = base_net.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs=base_net.input, outputs=outputs)

    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer, 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    return model