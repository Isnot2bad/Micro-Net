from read_data import *
from config import FLAGS
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import math

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.set_random_seed(FLAGS.seed)


def test_get_generator():
    """
    get_generator() test case

    :return:
    """
    # train_generator, image_datagen, label_datagen = get_generator()
    train_generator, steps_per_epoch = get_generator(
        subset="validation"  # "training" / "validation"
    )

    i = 0
    for image_batch, label_batch in iter(train_generator):
        print("image_batch.shape, label_batch.shape =", image_batch.shape, label_batch.shape)
        # image_batch = image_datagen.standardize(image_batch)
        # label_batch = label_datagen.standardize(label_batch)

        print(image_batch.dtype, label_batch.dtype)
        # print(image_batch[0])
        # print(label_batch[0])

        # # plt.figure(num='Sample', figsize=(8, 8))
        #
        # plt.subplot(2, 2, 1)
        # plt.title('Image')
        # plt.imshow(image_batch[0, ..., 0])
        #
        # plt.subplot(2, 2, 2)
        # plt.title('Label')
        # plt.imshow(label_batch[0, ..., 0])
        # plt.axis('off')
        #
        # plt.subplot(2, 2, 3)
        # plt.title('Image')
        # plt.imshow(image_batch[1, ..., 0])
        #
        # plt.subplot(2, 2, 4)
        # plt.title('Label')
        # plt.imshow(label_batch[1, ..., 0])
        # plt.axis('off')
        #
        # plt.show()
        i += 1
        if i == 5:
            break


if __name__ == "__main__":
    test_get_generator()
