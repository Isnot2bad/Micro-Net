"""
configurations
"""
import tensorflow as tf
import os


def _get_flags():
    """
    Get tensorflow configuration parameters

    :return: FLAG object
    """
    flags = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer("batch_size", 2, help="batch size")
    tf.app.flags.DEFINE_integer("epoch", 20, help="the number of epoch")
    tf.app.flags.DEFINE_float("lr", 1e-3, "learning rate")
    tf.app.flags.DEFINE_float("decay", flags.lr / flags.epoch, "decay of learning rate ")
    tf.app.flags.DEFINE_float("momentum", 0.9, "optimizer momentum parameter")
    tf.app.flags.DEFINE_float("validation_split", 0.1, "proportion of val dataset")

    tf.app.flags.DEFINE_boolean("use_gpus", False, "whether to use multi-GPU")
    tf.app.flags.DEFINE_boolean("use_reg_l2", False, "whether to use l2 regularization")
    tf.app.flags.DEFINE_float("l2", 5e-4, "l2 regularization parameter")

    tf.app.flags.DEFINE_integer("seed", 1, help="Random number seed")
    tf.app.flags.DEFINE_integer("nb_classes", 2, help="number of categories")
    tf.app.flags.DEFINE_integer("image_size", 5000, help="original image size")
    tf.app.flags.DEFINE_integer("sub_size", 500, help="cropped image size")
    tf.app.flags.DEFINE_integer("resize_size", 500, help="resized image size")
    tf.app.flags.DEFINE_integer("device_id", 1, help="The GPU to be used")
    tf.app.flags.DEFINE_integer("workers", 1, help="number of data pre-load threads")

    tf.app.flags.DEFINE_boolean("pre_trained", False, "whether to load the pre-training model")

    tf.app.flags.DEFINE_string("dataset_dir", os.path.expanduser("~/DATASET/AerialImageDataset"), "dataset directory")
    tf.app.flags.DEFINE_string("cropped", os.path.expanduser("cropped"), "folder for storing dataset after cropped")
    tf.app.flags.DEFINE_string("images_folder_name", "images", "Identifier used to identify whether the file is image")
    tf.app.flags.DEFINE_string("labels_folder_name", "gt", "Identifier used to identify whether the file is label")

    return flags


FLAGS = _get_flags()
