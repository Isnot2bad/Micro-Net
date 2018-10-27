"""
Micro-Net
"""
from keras.models import Model
from keras.layers import Input, Conv2D, Add, Concatenate, MaxPooling2D, Deconv2D


def get_fire_config(i, base_e, freq, squeeze_ratio, pct_3x3):
    """
    get configuration for fire module

    :param i: i-th Fire Module
    :param base_e: base number of e
    :param freq: The frequency of e_i updates
    :param squeeze_ratio: s_1x1 = squeeze_ratio * e_i
    :param pct_3x3: proportion of e_3x3 to e_i
    :return: parameters: e_i, s_1x1, e_1x1, e_3x3
    """
    assert isinstance(squeeze_ratio, float) and 0 <= squeeze_ratio <= 1, "Incorrect parameters"
    assert isinstance(pct_3x3, float) and 0 <= pct_3x3 <= 1, "Incorrect parameters"

    e_i = base_e * (2 ** (i // freq))
    s_1x1 = int(squeeze_ratio * e_i)
    e_3x3 = int(pct_3x3 * e_i)
    e_1x1 = e_i - e_3x3
    return e_i, s_1x1, e_1x1, e_3x3


def fire_module(inputs, fire_i, base_e, freq, squeeze_ratio, pct_3x3, dilation_rate,
                activation, kernel_initializer, data_format, use_bias=False, decoder=False):
    e_i, s_1x1, e_1x1, e_3x3 = get_fire_config(
        i=fire_i, base_e=base_e, freq=freq,
        squeeze_ratio=squeeze_ratio, pct_3x3=pct_3x3)
    if decoder:
        d = "decoder_"
    else:
        d = ""

    squeeze = Conv2D(
        s_1x1, (1, 1), use_bias=use_bias, activation=activation, kernel_initializer=kernel_initializer,
        padding='same', dilation_rate=dilation_rate, name=d + "fire%d_s1x1" % fire_i,
        data_format=data_format)(inputs)

    fire2_expand1 = Conv2D(
        e_1x1, (1, 1), use_bias=use_bias, activation=activation, kernel_initializer=kernel_initializer,
        padding='same', dilation_rate=dilation_rate, name=d + "fire%d_e1x1" % fire_i,
        data_format=data_format)(squeeze)

    fire2_expand2 = Conv2D(
        e_3x3, (3, 3), use_bias=use_bias, activation=activation, kernel_initializer=kernel_initializer,
        padding='same', dilation_rate=dilation_rate, name=d + "fire%d_e3x3" % fire_i,
        data_format=data_format)(squeeze)

    merge = Concatenate(axis=-1, name=d + "fire_merge%d" % fire_i)([fire2_expand1, fire2_expand2])
    return merge


def micro_net(nb_classes,
              base_e=64, freq=4, squeeze_ratio=0.25, pct_3x3=0.5,
              inputs_shape=(224, 224, 3),
              use_bias=False,
              data_format="channels_last",
              activation="relu",
              kernel_initializer="he_normal",
              ):
    """
    :param nb_classes: Number of categories
    :param base_e: base number of e
    :param freq: The frequency of e_i updates
    :param squeeze_ratio: s_1x1 = squeeze_ratio * e_i
    :param pct_3x3: proportion of e_3x3 to e_i
    :param inputs_shape : shape of input: (channel, cols, rows)
    :param use_bias: whether to use 'bias'
    :param data_format : data format: "channels_first" or "channels_last"
    :param activation : activation function
    :param kernel_initializer: weight initializer (in convolution layer)
    """
    inputs = Input(shape=inputs_shape)

    # Encoder
    conv1 = fire_module(inputs, 0, base_e, freq, squeeze_ratio, pct_3x3, 1,  # rate
                        activation, kernel_initializer, data_format)
    conv1 = fire_module(conv1, 1, base_e, freq, squeeze_ratio, pct_3x3, 1,  # rate
                        activation, kernel_initializer, data_format)
    conv1 = fire_module(conv1, 2, base_e, freq, squeeze_ratio, pct_3x3, 2,  # rate
                        activation, kernel_initializer, data_format)
    conv1 = fire_module(conv1, 3, base_e, freq, squeeze_ratio, pct_3x3, 3,  # rate
                        activation, kernel_initializer, data_format)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)

    conv2 = fire_module(pool1, 4, base_e, freq, squeeze_ratio, pct_3x3, 1,  # rate
                        activation, kernel_initializer, data_format)
    conv2 = fire_module(conv2, 5, base_e, freq, squeeze_ratio, pct_3x3, 1,  # rate
                        activation, kernel_initializer, data_format)
    conv2 = fire_module(conv2, 6, base_e, freq, squeeze_ratio, pct_3x3, 2,  # rate
                        activation, kernel_initializer, data_format)
    conv2 = fire_module(conv2, 7, base_e, freq, squeeze_ratio, pct_3x3, 3,  # rate
                        activation, kernel_initializer, data_format)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)

    conv3 = fire_module(pool2, 8, base_e, freq, squeeze_ratio, pct_3x3, 1,  # rate
                        activation, kernel_initializer, data_format)
    conv3 = fire_module(conv3, 9, base_e, freq, squeeze_ratio, pct_3x3, 1,  # rate
                        activation, kernel_initializer, data_format)
    conv3 = fire_module(conv3, 10, base_e, freq, squeeze_ratio, pct_3x3, 2,  # rate
                        activation, kernel_initializer, data_format)
    conv3 = fire_module(conv3, 11, base_e, freq, squeeze_ratio, pct_3x3, 3,  # rate
                        activation, kernel_initializer, data_format)

    # Decoder
    d_conv3 = fire_module(conv3, 10, base_e, freq, squeeze_ratio, pct_3x3, 3,  # rate
                          activation, kernel_initializer, data_format, decoder=True)
    d_conv3 = fire_module(d_conv3, 9, base_e, freq, squeeze_ratio, pct_3x3, 2,  # rate
                          activation, kernel_initializer, data_format, decoder=True)
    d_conv3 = fire_module(d_conv3, 8, base_e, freq, squeeze_ratio, pct_3x3, 1,  # rate
                          activation, kernel_initializer, data_format, decoder=True)
    up2 = Deconv2D(filters=128, kernel_size=(2, 2), strides=2, padding='same',
                   kernel_initializer=kernel_initializer, use_bias=use_bias)(d_conv3)

    added_conv3 = Add()([up2, conv2])
    d_conv2 = fire_module(added_conv3, 6, base_e, freq, squeeze_ratio, pct_3x3, 3,  # rate
                          activation, kernel_initializer, data_format, decoder=True)
    d_conv2 = fire_module(d_conv2, 5, base_e, freq, squeeze_ratio, pct_3x3, 2,  # rate
                          activation, kernel_initializer, data_format, decoder=True)
    d_conv2 = fire_module(d_conv2, 4, base_e, freq, squeeze_ratio, pct_3x3, 1,  # rate
                          activation, kernel_initializer, data_format, decoder=True)
    up1 = Deconv2D(filters=64, kernel_size=(2, 2), strides=2, padding='same',
                   kernel_initializer=kernel_initializer, use_bias=use_bias)(d_conv2)

    added_conv2 = Add()([up1, conv1])
    d_conv1 = fire_module(added_conv2, 2, base_e, freq, squeeze_ratio, pct_3x3, 3,  # rate
                          activation, kernel_initializer, data_format, decoder=True)
    d_conv1 = fire_module(d_conv1, 1, base_e, freq, squeeze_ratio, pct_3x3, 2,  # rate
                          activation, kernel_initializer, data_format, decoder=True)
    d_conv1 = fire_module(d_conv1, 0, base_e, freq, squeeze_ratio, pct_3x3, 1,  # rate
                          activation, kernel_initializer, data_format, decoder=True)

    # Classifier
    out_conv = Conv2D(nb_classes, kernel_size=1, use_bias=use_bias, activation='softmax')(d_conv1)

    # Model
    net = Model(inputs=inputs, outputs=out_conv)
    net.name = "micro_net_" + str(squeeze_ratio) + "_1123_1123_1123"
    return net
