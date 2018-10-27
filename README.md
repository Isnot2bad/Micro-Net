# Micro-Net


**This is the implementation of the models and code in paper:**
```
A MINIATURIZED SEMANTIC SEGMENTATION METHOD FOR REMOTE SENSING IMAGE

Shou-Yu Chen, Guang-Sheng Chen and Wei-Peng Jing
```
**Email: (Shou-Yu Chen)nefuchensy@163.com**


Instructions for use
---------

**Software and hardware:**

1. programming language: Python 3.6.
2. deep learning framework: Tensorflow 1.6 and Keras 2.0.
3. main hardware: Macbook Pro 16G, Intel Core i7 3.1GHz, NVIDIA 1080 eGPU (8G).

**Prepare the dataset:**

1. change 'dataset_dir' in ```config.py``` to your dataset root path.
2. In ```_test_utils.py```, change 'city_names_needed' in ```test_crop_dataset()```to the city list you want, \
and 'percent' to the percent of data amount in these cities you want to process, then, run this file to obtain\
dataset which model can train on it.

**Train the model**
1. you can choose one of 'unet' or 'micro_net' as model in ```__name__ == '__main__'``` in ```train.py```.
2. run ```python3 train.py``` to start training, Tensorboard log and model weight files will be automatically \
stored in the path defined by 'dataset_dir' in ```config.py```.

**Results**
1. run ```tensorboard --logdir=log``` in the path you save the log to start tensorboard.
2. open your browser and enter the 'http://localhost:6006' to observe the results.