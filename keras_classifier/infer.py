import os
import sys
import tensorflow as tf
import numpy as np
import requests
import matplotlib.pyplot as plt
from PIL import Image

labels_map_directory = os.path.dirname(os.path.dirname(__file__))
sys.path.append(labels_map_directory)
from labels_map import labels_map

ankle_boot_sample_image = 'https://raw.githubusercontent.com/MicrosoftDocs/tensorflow-learning-path/main/intro-keras/predict-image.png'

def get_data(argv):
    """
    Load image to classify either from first argument or input.
    If neither is provided a sample image is used
    """
    url = None

    if len(argv) != 2:
        url = input('\nUrl to the image you want to classify: ')
    else:
        url = argv[1]
        
    if not url:
        url = ankle_boot_sample_image

    with Image.open(requests.get(url, stream=True).raw) as image:
        X = np.asarray(image, dtype=np.float32).reshape((-1, 28, 28)) / 255
    return X

def display_prediction(image, label_key):
    """
    Graphically display image and its predicted label as title
    """
    plt.figure()
    plt.axis('off')
    plt.title(labels_map[label_key])
    plt.imshow(image.squeeze(), cmap='gray')
    plt.show()

def main(argv):
    model = tf.keras.models.load_model('output/kclassifier')
    X = get_data(argv)

    print("\nPredicting:")
    predicted_vector = model.predict(X)
    predicted_idx = np.argmax(predicted_vector)
    display_prediction(X, predicted_idx)

    print("\nProbabilties for each class:")
    probs = tf.nn.softmax(predicted_vector.reshape((-1)))
    for i,p in enumerate(probs):
        print(f"{labels_map[i]} -> {p:.3f}")

if __name__ == "__main__":
    main(sys.argv)
