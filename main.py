import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import data as tf_data
from keras.src.utils import load_img, img_to_array
from keras.src.optimizers.adam import Adam
from keras.src.optimizers.schedules import ExponentialDecay
from keras.src.losses.losses import CategoricalCrossentropy
from keras.src.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras_cv.src.layers import Equalization
from keras.src.models import Model
from keras.src.layers import RandomZoom, RandomBrightness, RandomFlip, RandomRotation
from keras.src.utils.image_dataset_utils import image_dataset_from_directory
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from keras.src.applications.efficientnet import EfficientNetB0
from keras.src.applications.densenet import DenseNet169

BATCH_SIZE = 32
# effnet -> 224, custom -> 128, densenet -> 64
IMG_HEIGHT = 224
IMG_WIDTH = 224
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 100

SAVE_PROGRESS = True

saved_model_url = "../effnet.keras"
training_dataset_url = "C:/Fakultet/Semestar 6/Racunarska Inteligencija/Projekat/Data/Training"
testing_dataset_url = "C:/Fakultet/Semestar 6/Racunarska Inteligencija/Projekat/Data/Testing"

prediction_image_url = "C:/Fakultet/Semestar 6/Racunarska Inteligencija/Projekat/no_tumor.png"

labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

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


def build_model():
    model = Sequential([
        keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(128, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(256, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(512, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(NUM_CLASSES, activation='softmax', name='outputs')
    ])
    return model


def use_pretrained_model_efficient():

    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


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

    visualize_augmentation(train_dataset)
    visualize_data(train_dataset)
    train_dataset = train_dataset.map(lambda img, label: (data_augmentation(img), label),
                                      num_parallel_calls=tf_data.AUTOTUNE
                                      )
    train_dataset = train_dataset.prefetch(tf_data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf_data.AUTOTUNE)

    # Creating a model
    # Change function call to select model
    generated_model = build_model()

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


def visualize_augmentation(dataset):
    plt.figure(figsize=(10, 10))
    for images, _ in dataset.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(augmented_images[i]).astype("uint8"), "gray")
            plt.axis("off")
    plt.show()


def visualize_data(dataset):
    plt.figure(figsize=(9, 9))
    for images, labels in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(images[i]).astype("uint8"), "gray")
            plt.title(class_names[np.argmax(labels[i])])
            plt.axis("off")
    plt.show()


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


def evaluate_and_print_statistics(model=None):
    # Get the true labels and predictions
    if model is None:
        model = keras.src.models.model.saving_api.load_model(saved_model_url)
    test_dataset = image_dataset_from_directory(
        testing_dataset_url,
        shuffle=True,
        label_mode='categorical',
        color_mode='grayscale',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    true_labels = []
    predictions = []

    for images, labels in test_dataset:
        preds = model.predict(images)
        true_labels.extend(np.argmax(labels.numpy(), axis=1))
        predictions.extend(np.argmax(preds, axis=1))

    true_labels = np.array(true_labels)
    predictions = np.array(predictions)

    cm = confusion_matrix(true_labels, predictions)
    plot_confusion_matrix(cm, list(class_names.values()))

    print("\nClassification Report:\n")
    print(classification_report(true_labels, predictions, target_names=list(class_names.values())))


def make_prediction(trained_model=None):
    img_to_predict = load_img(
        prediction_image_url,
        color_mode='grayscale',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
    )
    input_arr = img_to_array(img_to_predict)
    input_arr = np.array([input_arr])
    if trained_model is None:
        trained_model = keras.src.models.model.saving_api.load_model(saved_model_url)
    predictions = trained_model.predict(input_arr)
    score = tf.nn.softmax(predictions[0])
    print(
        f"This image most likely belongs to {class_names[np.argmax(score)]} with a {100 * np.max(score):.2f} percent confidence.")
    print(predictions)

if __name__ == "__main__":
    print(tf.keras.__version__)

    # model = train_and_evaluate()
    # evaluate_and_print_statistics()
    make_prediction()