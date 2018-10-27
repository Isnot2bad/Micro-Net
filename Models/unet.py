"""
UNet
"""
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Cropping2D, Deconv2D


def unet(inputs_shape, nb_classes, use_bias=False, activation='relu', kernel_initializer='he_normal'):
    inputs = Input(shape=inputs_shape)

    # Encoder
    conv1 = Conv2D(64, 3, use_bias=use_bias, activation=activation, padding='same',
                   kernel_initializer=kernel_initializer)(inputs)
    conv1 = Conv2D(64, 3, use_bias=use_bias, activation=activation, padding='same',
                   kernel_initializer=kernel_initializer)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)

    conv2 = Conv2D(128, 3, use_bias=use_bias, activation=activation, padding='same',
                   kernel_initializer=kernel_initializer)(pool1)
    conv2 = Conv2D(128, 3, use_bias=use_bias, activation=activation, padding='same',
                   kernel_initializer=kernel_initializer)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)

    conv3 = Conv2D(256, 3, use_bias=use_bias, activation=activation, padding='same',
                   kernel_initializer=kernel_initializer)(pool2)
    conv3 = Conv2D(256, 3, use_bias=use_bias, activation=activation, padding='same',
                   kernel_initializer=kernel_initializer)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3)

    conv4 = Conv2D(512, 3, use_bias=use_bias, activation=activation, padding='same',
                   kernel_initializer=kernel_initializer)(pool3)
    conv4 = Conv2D(512, 3, use_bias=use_bias, activation=activation, padding='same',
                   kernel_initializer=kernel_initializer)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv4)

    conv5 = Conv2D(1024, 3, use_bias=use_bias, activation=activation, padding='same',
                   kernel_initializer=kernel_initializer)(pool4)
    conv5 = Conv2D(1024, 3, use_bias=use_bias, activation=activation, padding='same',
                   kernel_initializer=kernel_initializer)(conv5)

    # Decoder
    up6 = Deconv2D(filters=512, kernel_size=(2, 2), strides=2, padding='same',
                   kernel_initializer=kernel_initializer, use_bias=use_bias)(conv5)
    cropped_up6 = Cropping2D(
        cropping=((0, 1),
                  (0, 1))
    )(up6)
    merge6 = Concatenate(axis=-1)([conv4, cropped_up6])
    conv6 = Conv2D(512, 3, use_bias=use_bias, activation=activation, padding='same',
                   kernel_initializer=kernel_initializer)(merge6)
    conv6 = Conv2D(512, 3, use_bias=use_bias, activation=activation, padding='same',
                   kernel_initializer=kernel_initializer)(conv6)

    up7 = Deconv2D(filters=256, kernel_size=(2, 2), strides=2, padding='same',
                   kernel_initializer=kernel_initializer, use_bias=use_bias)(conv6)
    cropped_up7 = Cropping2D(
        cropping=((0, 1),
                  (0, 1))
    )(up7)
    merge7 = Concatenate(axis=3)([conv3, cropped_up7])
    conv7 = Conv2D(256, 3, use_bias=use_bias, activation=activation, padding='same',
                   kernel_initializer=kernel_initializer)(merge7)
    conv7 = Conv2D(256, 3, use_bias=use_bias, activation=activation, padding='same',
                   kernel_initializer=kernel_initializer)(conv7)

    up8 = Deconv2D(filters=128, kernel_size=(2, 2), strides=2, padding='same',
                   kernel_initializer=kernel_initializer, use_bias=use_bias)(conv7)
    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = Conv2D(128, 3, use_bias=use_bias, activation=activation, padding='same',
                   kernel_initializer=kernel_initializer)(merge8)
    conv8 = Conv2D(128, 3, use_bias=use_bias, activation=activation, padding='same',
                   kernel_initializer=kernel_initializer)(conv8)

    up9 = Deconv2D(filters=64, kernel_size=(2, 2), strides=2, padding='same',
                   kernel_initializer=kernel_initializer, use_bias=use_bias)(conv8)
    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9 = Conv2D(64, 3, use_bias=use_bias, activation=activation, padding='same',
                   kernel_initializer=kernel_initializer)(merge9)
    conv9 = Conv2D(64, 3, use_bias=use_bias, activation=activation, padding='same',
                   kernel_initializer=kernel_initializer)(conv9)

    conv10 = Conv2D(nb_classes, kernel_size=1, use_bias=use_bias, activation='softmax')(conv9)

    net = Model(inputs=inputs, outputs=conv10)
    net.name = "unet"
    return net
