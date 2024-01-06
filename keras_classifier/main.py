from classifier import ClassifierNetwork
import tensorflow as tf
import random
import matplotlib.pyplot as plt

labels_map = {
    0: 'T-Shirt',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle Boot'
}

(training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

def display_sample(images, labels):
    cols = 3
    rows = 3
    figure = plt.figure(figsize=(6, 6))
    
    for i in range(1, cols * rows + 1):
        sample_idx = random.randint(0, len(images))
        image = images[sample_idx]
        label_key = labels[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.axis('off')
        plt.imshow(image.squeeze(), 'gray')
        plt.title(labels_map[label_key])
    plt.show()

def explore_data(images, labels):
    display_sample(images, labels)
    print()
    print('Image:\n', images[0])
    print()
    print('Label:\n', labels_map[labels[0]])

def main():
    explore_data(training_images, training_labels)

if __name__ == "__main__":
    main()
