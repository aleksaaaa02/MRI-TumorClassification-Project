import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.src.utils import load_img, img_to_array
from keras.src.utils.image_dataset_utils import image_dataset_from_directory
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from keras.src.applications.efficientnet import EfficientNetB0
from keras.src.applications.densenet import DenseNet169

MODEL = 'densenet'
BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE

DIMENSIONS = {
    'effnet': 64,
    'densenet': 64,
    'custom': 128
}

IMG_HEIGHT = DIMENSIONS[MODEL]
IMG_WIDTH = DIMENSIONS[MODEL]
GRAY_SCALE_ON = 'rgb' if MODEL == 'densenet' or MODEL == 'effnet' else 'grayscale'

saved_model_url = "../densenet169_64_64.keras"
testing_dataset_url = "C:/Fakultet/Semestar 6/Racunarska Inteligencija/Projekat/Data/Testing"
prediction_image_url = "C:/Fakultet/Semestar 6/Racunarska Inteligencija/Projekat/no_tumor.png"

class_names = {
    0: 'glioma_tumor',
    1: 'meningioma_tumor',
    2: 'no_tumor',
    3: 'pituitary_tumor'
}

NUM_CLASSES = 4

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
        color_mode=GRAY_SCALE_ON,
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


if __name__ == '__main__':
    evaluate_and_print_statistics()

