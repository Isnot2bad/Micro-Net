"""
Load dataset
"""
import tensorflow as tf
from config import FLAGS
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)


def get_generator(subset):
    """
    Get dataset generator

    :param subset: The dataset to be returned. "training" or "validation"
    :return:
    """
    assert subset == "training" or subset == "validation"

    image_datagen = ImageDataGenerator(
        horizontal_flip=True,
        validation_split=FLAGS.validation_split,
        rescale=1. / 255.,
    )
    image_generator = image_datagen.flow_from_directory(
        directory=
        os.path.join(
            FLAGS.dataset_dir, FLAGS.cropped, "train", FLAGS.images_folder_name,
        ),
        target_size=(FLAGS.resize_size, FLAGS.resize_size),
        class_mode=None,
        color_mode="rgb",
        batch_size=FLAGS.batch_size,
        seed=FLAGS.seed,
        subset=subset,
        shuffle=False if subset == "validation" else True,
    )

    label_datagen = ImageDataGenerator(
        horizontal_flip=True,
        validation_split=FLAGS.validation_split,
    )
    label_generator = label_datagen.flow_from_directory(
        directory=
        os.path.join(
            FLAGS.dataset_dir, FLAGS.cropped, "train", FLAGS.labels_folder_name,
        ),
        target_size=(FLAGS.resize_size, FLAGS.resize_size),
        class_mode=None,
        color_mode="grayscale",
        batch_size=FLAGS.batch_size,
        seed=FLAGS.seed,
        subset=subset,
        shuffle=False if subset == "validation" else True,
    )

    train_generator = zip(image_generator, label_generator)
    steps_per_epoch = len(label_generator)
    print("steps_per_epoch =", steps_per_epoch)
    return train_generator, steps_per_epoch
