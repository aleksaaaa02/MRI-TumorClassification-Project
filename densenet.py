import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import data as tf_data
from keras.src.optimizers.adam import Adam
from keras.src.losses.losses import CategoricalCrossentropy
from keras.src.layers import Dense, GlobalAveragePooling2D
from keras.src.models import Model
from keras.src.layers import RandomZoom, RandomBrightness, RandomFlip, RandomRotation
from keras.src.utils.image_dataset_utils import image_dataset_from_directory
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.applications.densenet import DenseNet169

BATCH_SIZE = 32
IMG_HEIGHT = 64
IMG_WIDTH = 64
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 50

SAVE_PROGRESS = True

saved_model_url = "../effnet.keras"
training_dataset_url = "C:/Fakultet/Semestar 6/Racunarska Inteligencija/Projekat/Data/Training"

class_names = {
    0: 'glioma_tumor',
    1: 'meningioma_tumor',
    2: 'no_tumor',
    3: 'pituitary_tumor'
}

NUM_CLASSES = 4

data_augmentation_layers = [
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.2),
    RandomBrightness(factor=0.2)
]

def use_pretrained_model_densenet():
    base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def train_and_evaluate():
    train_dataset, val_dataset = image_dataset_from_directory(
        training_dataset_url,
        validation_split=0.2,
        subset="both",
        label_mode='categorical',
        color_mode='grayscale',
        shuffle=True,
        seed=1337,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    train_dataset = train_dataset.map(lambda img, label: (data_augmentation(img), label),
                                      num_parallel_calls=tf_data.AUTOTUNE
                                      )
    train_dataset = train_dataset.prefetch(tf_data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf_data.AUTOTUNE)

    # Creating a model
    # Change function call to select model
    generated_model = use_pretrained_model_densenet()

    generated_model.compile(optimizer=Adam(0.0001),
                            loss=CategoricalCrossentropy(from_logits=False),
                            metrics=['accuracy'])

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    callbacks = [early_stopping_callback]

    if SAVE_PROGRESS:
        checkpoint_callback = ModelCheckpoint(filepath='best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
        callbacks.append(checkpoint_callback)

    generated_model.summary()
    history = generated_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS,
            callbacks=callbacks)

    plot_training_history(history)
    return generated_model


def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images


if __name__ == "__main__":
    print(tf.keras.__version__)

    model = train_and_evaluate()