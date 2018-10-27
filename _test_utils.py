from utils import *


def test_get_city_and_no():
    """
    get_city_and_no() test case

    :return:
    """
    get_city_and_no(image_path="/Users/chen/TIFF_DATA/AerialImageDataset/train/images/austin26.png")


def test_crop_dataset():
    """
    crop_dataset() test case

    :return:
    """
    crop_dataset(output_dir=os.path.join(FLAGS.dataset_dir, FLAGS.cropped),
                 sub_size=FLAGS.sub_size,
                 stride=FLAGS.sub_size // 2,
                 city_names_needed=["austin"],
                 percent=0.5)


if __name__ == "__main__":
    # test_get_city_and_no()
    test_crop_dataset()
