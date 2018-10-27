"""
test for unet
"""

from Models import unet
from config import FLAGS


def test_unet():
    """
    unet() test case
    :return:
    """
    model = unet(inputs_shape=(500, 500, 3),
                 nb_classes=FLAGS.nb_classes)
    model.summary()


if __name__ == '__main__':
    test_unet()
