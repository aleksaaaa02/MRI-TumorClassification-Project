import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import data as tf_data
from keras.src.utils import load_img, img_to_array
from keras.src.optimizers.adam import Adam
from keras.src.losses.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from keras.src.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras_cv.src.layers import RandomFlip, RandomRotation, Rescaling, Equalization, Posterization
from keras.src.utils.image_dataset_utils import image_dataset_from_directory
from keras.src.callbacks import ModelCheckpoint, EarlyStopping


BATCH_SIZE = 32
IMG_HEIGHT = 180
IMG_WIDTH = 180
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 25

SAVE_PROGRESS = True

training_dataset_url = "C:\\Fakultet\\Semestar 6\\Racunarska Inteligencija\\Projekat\\Data\\Training"
testing_dataset_url = "C:\\Fakultet\\Semestar 6\\Racunarska Inteligencija\\Projekat\\Data\\Testing"

prediction_image_url = "C:\\Fakultet\\Semestar 6\\Racunarska Inteligencija\\Projekat\\glioma_test.jpeg"

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
    Equalization([0, 255], 256),
    # Posterization(value_range=[0, 160], bits=7),
    Rescaling(scale=1./255)
]


def build_model():
    model = Sequential([
        keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        Conv2D(16, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(128, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(NUM_CLASSES, activation='softmax', name='outputs')
    ])
    return model


def train_and_evaluate():

    train_dataset = image_dataset_from_directory(
        training_dataset_url,
        label_mode='categorical',
        color_mode='grayscale',
        shuffle=True,
        seed=1337,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    val_dataset = image_dataset_from_directory(
        testing_dataset_url,
        label_mode='categorical',
        color_mode='grayscale',
        shuffle=True,
        seed=1337,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    visualize_augmentation(train_dataset)
    visualize_data(train_dataset)
    train_dataset = train_dataset.map(lambda img, label: (data_augmentation(img), label),
                                      num_parallel_calls=tf_data.AUTOTUNE
                                      )
    train_dataset = train_dataset.prefetch(tf_data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf_data.AUTOTUNE)

    # Creating a model
    generated_model = build_model()

    generated_model.compile(optimizer=Adam(learning_rate=0.0001),
                            loss=CategoricalCrossentropy(from_logits=False),
                            metrics=['accuracy'])

    # early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    # callbacks = [early_stopping_callback]
    # if SAVE_PROGRESS:
    #     checkpoint_callback = ModelCheckpoint(filepath='best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
    #     callbacks.append(checkpoint_callback)

    generated_model.summary()
    callbacks = []
    history = generated_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS,
            callbacks=callbacks)

    if SAVE_PROGRESS:
        generated_model.save('my_model.keras')

    # plot_training_history(history)

    return generated_model


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


def evaluate_test_dataset(trained_model=None):
    img_to_predict = load_img(
        prediction_image_url,
        color_mode='grayscale',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
    )
    input_arr = img_to_array(img_to_predict)
    input_arr = np.array([input_arr])

    if trained_model is None:
        trained_model = keras.src.models.model.saving_api.load_model('my_model.keras')
    # testing_dataset = testing_dataset.prefetch(buffer_size=AUTOTUNE)

    # Need changing, should choose a picture than do a prediction on it
    predictions = trained_model.predict(input_arr)

    score = tf.nn.softmax(predictions[0])
    print(
        f"This image most likely belongs to {class_names[np.argmax(score)]} with a {100 * np.max(score):.2f} percent confidence.")
    print(predictions)


def filter_corrupted_images():
    num_skipped = 0
    for folder_name in (""):
        pass


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images


def visualize_augmentation(dataset):
    plt.figure(figsize=(10, 10))
    for images, _ in dataset.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(augmented_images[i]).astype("uint8"))
            plt.axis("off")
    plt.show()


def visualize_data(dataset):
    plt.figure(figsize=(9, 9))
    for images, labels in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(images[i]).astype("uint8"))
            plt.title(class_names[np.argmax(labels[i])])
            plt.axis("off")
    plt.show()


if __name__ == "__main__":
    print("Hello World from my project!")
    print(tf.keras.__version__)

    # model = train_and_evaluate()

    evaluate_test_dataset()
