"""
Utils
"""

import os
from config import FLAGS
import tensorflow as tf
import glob
from scipy import misc
from tqdm import tqdm
import keras

CITYS = {
    1: "austin",
    2: "chicago",
    3: "kitsap",
    4: "tyrol-w",
    5: "vienna",
}  # city list


class TrainValTensorBoard(keras.callbacks.TensorBoard):
    """
    Customized callback function for tensorboard that can output train/val at the same time
    """

    def __init__(self, log_dir='./logs', **kwargs):
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


def get_city_and_no(image_path):
    """
    Get the city name and serial number of the file based on the incoming image path

    :param image_path: image path
    :return: (city name, serial number)
    """
    image_path = image_path.split("/")[-1].split(".")[0]
    city_name = ""
    no = -1
    has_found = False
    for value in CITYS.values():
        if image_path.startswith(value):
            city_name = value
            no = int(image_path.split(city_name)[-1])
            has_found = True
            break
    assert has_found, "City not found!"
    return city_name, no


def crop_dataset(output_dir, sub_size, stride, city_names_needed, percent):
    """
    Crop the dataset

    :param output_dir: Output path
    :param sub_size: Sub-image size
    :param stride: stride
    :param city_names_needed: List of city names to be cropped
    :param percent: Percentage of original images for each city to be cropped
    :return: None
    """
    assert isinstance(city_names_needed, list)
    assert isinstance(percent, float)

    if not os.path.exists(os.path.join(output_dir, "train", FLAGS.images_folder_name, FLAGS.images_folder_name)):
        os.makedirs(os.path.join(output_dir, "train", FLAGS.images_folder_name, FLAGS.images_folder_name))
    if not os.path.exists(os.path.join(output_dir, "train", FLAGS.labels_folder_name, FLAGS.labels_folder_name)):
        os.makedirs(os.path.join(output_dir, "train", FLAGS.labels_folder_name, FLAGS.labels_folder_name))

    image_paths = glob.glob(os.path.join(FLAGS.dataset_dir, "train", FLAGS.images_folder_name, "*.png"))
    for image_path in tqdm(image_paths):
        label_path = image_path.replace("images", "gt")
        city_name, city_no = get_city_and_no(image_path)

        if not (city_name in city_names_needed and city_no < int(36 * percent)):
            continue
        print("city_name = %s, city_no = %d" % (city_name, city_no))

        image = misc.imread(image_path)
        label = misc.imread(label_path)

        label[label == 255] = 1
        h, w = label.shape

        sub_counter = 1
        for h_pixel in range(0, h - sub_size + 1, stride):
            for w_pixel in range(0, w - sub_size + 1, stride):
                sub_image = image[h_pixel:h_pixel + sub_size, w_pixel:w_pixel + sub_size, :]
                sub_label = label[h_pixel:h_pixel + sub_size, w_pixel:w_pixel + sub_size]
                sub_image_path = os.path.join(
                    output_dir, "train", FLAGS.images_folder_name, FLAGS.images_folder_name,
                    city_name + "-%d-%d.png" % (city_no, sub_counter))
                sub_label_path = sub_image_path.replace("images", "gt")

                misc.imsave(name=sub_image_path, arr=sub_image)
                misc.imsave(name=sub_label_path, arr=sub_label)
                sub_counter += 1
