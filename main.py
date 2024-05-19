import numpy as np
import tensorflow as tf
import PIL
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


BATCH_SIZE = 64
IMG_HEIGHT = 320
IMG_WIDTH = 320
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 5

class_names = ['glioma_tumor',
                'no_tumor',
                'meningioma_tumor',
                'pituitary_tumor']


def build_model(num_classes):
    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax', name='outputs')
    ])
    return model


def train_and_evaluate(training_dataset_url, testing_dataset_url):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_dataset = train_datagen.flow_from_directory(
        training_dataset_url,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        subset='training',
        class_mode='sparse',
        seed=123
    )

    val_dataset = train_datagen.flow_from_directory(
        training_dataset_url,
        color_mode='rgb',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        subset='validation',
        class_mode='sparse',
        seed=123,
        interpolation='bilinear'
    )

    num_classes = len(class_names)

    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)


    # Creating a model
    model = build_model(num_classes)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    checkpoint_callback = ModelCheckpoint(filepath='best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.summary()
    history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS,
            callbacks=[checkpoint_callback, early_stopping_callback])
    model.save('my_model.keras')
    plot_training_history(history)

    return model


def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

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


def evaluate_test_dataset(test_dataset_url, model=None):
    testing_dataset = tf.keras.utils.image_dataset_from_directory(
        testing_dataset_url,
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        shuffle=False,
        interpolation='bilinear',
    )
    if model is None:
        model = tf.keras.models.load_model('my_model.keras')
    test_dataset = testing_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test accuracy: {test_acc:.2f} | Test loss: {test_loss:.2f}")


if __name__ == "__main__":
    print("Hello World from my project!")
    print(tf.keras.__version__)

    training_dataset_url = "C:\Fakultet\Semestar 6\Racunarska Inteligencija\Projekat\Data\Training"
    testing_dataset_url = "C:\Fakultet\Semestar 6\Racunarska Inteligencija\Projekat\Data\Testing"

    model = train_and_evaluate(training_dataset_url, testing_dataset_url)
    evaluate_test_dataset(testing_dataset_url)