from keras.src.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
from sps.config import Config

def get_mnist_data():
    (train_data, train_label), (test_data, test_label) = mnist.load_data()
    train_data = train_data[:Config.TRAIN_SIZE]
    train_label = train_label[:Config.TRAIN_SIZE]
    test_data = test_data[:Config.TEST_SIZE]
    test_label = test_label[:Config.TEST_SIZE]
    train_q = train_data.astype(np.uint8)
    test_q = test_data.astype(np.uint8)
    if Config.INVERT:
        train_q = 255 - train_q
        test_q = 255 - test_q
    return train_q, train_label, test_q, test_label


def show_digit(x, y, train = False):
    x_np = x
    if not isinstance(x, np.ndarray):
        x_np = x.to_numpy()
    y_np = y.to_numpy().ravel()  # make labels 1D

    nrows = 3
    ncols = 10
    if train and Config.TRAIN_SIZE < 30:
        nrows = 2
        ncols = int((Config.TRAIN_SIZE - 1) / 2)
    if train and Config.TRAIN_SIZE < 30:
        nrows = 2
        ncols = int((Config.TRAIN_SIZE - 1) / 2)

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3))



    for row in range(nrows):
        for col in range(ncols):
            idx = row * ncols + col
            image = x_np[idx].reshape(8, 8)

            ax = axes[row, col]
            ax.imshow(image, cmap="gray_r", interpolation="nearest")
            ax.set_title(f"Label: {y_np[idx]}")
            ax.axis("off")

    plt.tight_layout()
    plt.show()