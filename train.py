from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import random
from Models import unet, micro_net
from config import FLAGS
import os
from utils import TrainValTensorBoard
from keras import losses
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from read_data import get_generator
from Metrics import miou
from keras.utils import multi_gpu_model
import numpy as np

# Automatic GPU memory allocation
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.device_id)
random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)
np.random.seed(FLAGS.seed)

if __name__ == '__main__':
    # Create model
    if FLAGS.use_gpus:
        with tf.device('/cpu:0'):
            # _model = unet(inputs_shape=(FLAGS.resize_size, FLAGS.resize_size, 3), nb_classes=FLAGS.nb_classes)
            _model = micro_net(nb_classes=FLAGS.nb_classes,
                               inputs_shape=(FLAGS.resize_size, FLAGS.resize_size, 3))

    else:
        # _model = unet(inputs_shape=(FLAGS.resize_size, FLAGS.resize_size, 3), nb_classes=FLAGS.nb_classes)
        _model = micro_net(nb_classes=FLAGS.nb_classes,
                           inputs_shape=(FLAGS.resize_size, FLAGS.resize_size, 3))

    _model.summary()

    # Whether to load a trained model
    if FLAGS.pre_trained:
        _model.load_weights(os.path.join(FLAGS.dataset_dir, _model.name + ".hdf5"))

    if FLAGS.use_gpus:
        model = multi_gpu_model(model=_model, gpus=2)
    else:
        model = _model

    # Optimizer
    sgd = SGD(lr=FLAGS.lr, decay=FLAGS.decay, momentum=FLAGS.momentum, nesterov=True)

    model.compile(optimizer=sgd,
                  loss=losses.sparse_categorical_crossentropy,  # losses.categorical_crossentropy,
                  metrics=['acc', miou])

    # checkpoint callback function
    checkpoint_callback = ModelCheckpoint(os.path.join(FLAGS.dataset_dir, _model.name + ".hdf5"),
                                          monitor='val_loss',
                                          verbose=1,
                                          save_best_only=True)
    # tensorboard log callback function
    tb_callback = TrainValTensorBoard(log_dir=os.path.join(FLAGS.dataset_dir, "log_" + _model.name))

    # Dataset generator
    train_generator, train_steps_per_epoch = get_generator(subset="training")
    val_generator, val_steps_per_epoch = get_generator(subset="validation")

    # Start training
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_steps_per_epoch,
        epochs=FLAGS.epoch,
        verbose=1,
        validation_data=val_generator,
        validation_steps=val_steps_per_epoch,
        callbacks=[checkpoint_callback, tb_callback],
        workers=FLAGS.workers * 2 if FLAGS.use_gpus else FLAGS.workers,
    )
