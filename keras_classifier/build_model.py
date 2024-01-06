import os
import sys
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from classifier import ClassifierNetwork

labels_map_directory = os.path.dirname(os.path.dirname(__file__))
sys.path.append(labels_map_directory)
from labels_map import labels_map

# Load fashion MNIST data from keras' datasets
(training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

def display_sample(images, labels, rows = 3, cols = 3):
    """
    Graphically display a sample of images with their labels in a grid of rows*cols size
    """
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
    """
    Provide a simple exploration of data consisting of:
    - graphical displaying of images and labels
    - console displaying of first image in *images*
    - console displaying of its label from *labels*
    """
    display_sample(images, labels)
    print()
    print('Image:\n', images[0])
    print()
    print('Label:\n', labels_map[labels[0]])
    
def build_datasets(batch_size):
    """
    Build one dataset for model training and another for testing from loaded data
    Here are some specifications of these datasets:
    - Images contain floating point values between 0 and 1 instead of unsigned int values between 0 and 255
    - Their batch size is set to 64 so that iterating over these datasets provides this number of items (only one batch is stored in memory)
    """
    train_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    
    train_dataset = train_dataset.map(lambda image, label: (float(image) / 255.0, label))
    test_dataset = test_dataset.map(lambda image, label: (float(image) / 255.0, label))
    # image, label_key = train_dataset.as_numpy_iterator().next()
    # print(image)
    # print(labels_map[label_key])
    
    train_dataset = train_dataset.batch(batch_size).shuffle(len(train_dataset))
    test_dataset = test_dataset.batch(batch_size).shuffle(len(test_dataset))
    return train_dataset, test_dataset

def compile_model(model, learning_rate = 0.01):
    """
    Compile model with SparseCategoricalCrossentropy as loss function and Stochastic Gradient Descent as optimizer
    """
    # *from_logits*'s value specify that a softmax must be applied to y_hat before computing
    # the loss as categorical cross entropy requires a probability distribution (values between 0 and 1, adding up to 1)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    optimizer = tf.keras.optimizers.SGD(learning_rate)

    # Metrics to report during training
    metrics = ['accuracy']

    model.compile(optimizer, loss_fn, metrics)

def main():
    batch_size = 64
    train_dataset, test_dataset = build_datasets(batch_size)
    model = ClassifierNetwork()
    epochs = 5
    learning_rate = 0.1

    print('\nCompiling:')
    compile_model(model, learning_rate)

    print('\nFitting:')
    model.fit(train_dataset, epochs=epochs)

    print('\nEvaluating:')
    (test_loss, test_accuracy) = model.evaluate(test_dataset)
    print(f"\nTest Accuracy: {test_accuracy * 100:>0.1f}%, Test Loss: {test_loss:>0.8f}")

    model.save('output/kclassifier')

if __name__ == "__main__":
    main()
