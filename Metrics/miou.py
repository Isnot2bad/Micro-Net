"""
mIOU
"""

from keras import backend as K
from config import FLAGS
from tensorflow.contrib.metrics import confusion_matrix
import tensorflow as tf


def miou(y_true, y_pred, n_classes=FLAGS.nb_classes):
    """
    Calculate mIOU

    :param y_true: labels of one batch, shape=(batch_size, H, W, n_classes) (one-hot format)
    :param y_pred: predicts of one batch, shape=(batch_size, H, W, n_classes)
    :param n_classes: Number of categories

    :return: Average of the IOU for each category
    """

    class_label = K.reshape(y_true, shape=[-1])

    class_pred = K.argmax(y_pred, -1)
    class_pred = K.reshape(class_pred, shape=[-1])

    confusion_metrics = confusion_matrix(labels=class_label,
                                         predictions=class_pred,
                                         num_classes=n_classes)

    denominator = tf.cast(n_classes, dtype=tf.float32)
    avg_iou = 0.

    for class_index in range(0, n_classes):
        sum_of_one_class = K.sum(confusion_metrics[class_index])

        denominator = tf.cond(
            tf.equal(sum_of_one_class, 0),
            lambda: tf.cast(denominator - 1, dtype=tf.float32),
            lambda: tf.cast(denominator, dtype=tf.float32)
        )

        # | X ∩ Y |
        intersection = confusion_metrics[class_index, class_index]
        # | X | + | Y |
        union = K.sum(confusion_metrics[class_index, :]) + K.sum(confusion_metrics[:, class_index]) - intersection

        iou_one_class = tf.cond(
            tf.equal(intersection, 0),
            lambda: 0.,
            lambda: tf.cast(intersection / union, dtype=tf.float32)
        )
        avg_iou += iou_one_class

    return tf.cond(
        tf.equal(denominator, 0),
        lambda: tf.cast(0, dtype=tf.float32),
        lambda: tf.cast(avg_iou / denominator, dtype=tf.float32)
    )
