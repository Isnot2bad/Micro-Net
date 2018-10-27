"""
test for micro-net
"""
from Models.micro_net import *


def test_micro_net():
    """
    micro_net() test case

    :return:
    """
    model = micro_net(nb_classes=2, inputs_shape=(500, 500, 3))
    model.summary()


if __name__ == '__main__':
    test_micro_net()
